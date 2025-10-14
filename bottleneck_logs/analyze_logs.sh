#!/bin/bash
# Log analysis script for bottleneck investigation
# Usage: ./analyze_logs.sh <path_to_orchestrator.log>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_orchestrator_log> [path_to_inference_log]"
    exit 1
fi

ORCH_LOG="$1"
INF_LOG="${2:-}"
OUTPUT_DIR="bottleneck_logs/analysis_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "===== Bottleneck Analysis Report ====="
echo "Generated: $(date)"
echo "Orchestrator Log: $ORCH_LOG"
echo "Inference Log: ${INF_LOG:-N/A}"
echo ""

# Extract weight update operations
echo ">> Extracting weight update operations..."
grep '\[weights\]' "$ORCH_LOG" > "$OUTPUT_DIR/weights_raw.log" 2>/dev/null || echo "No [weights] entries found"

# Extract checkpoint operations
echo ">> Extracting checkpoint operations..."
grep '\[ckpt\]' "$ORCH_LOG" > "$OUTPUT_DIR/ckpt_raw.log" 2>/dev/null || echo "No [ckpt] entries found"

# Extract generation operations
echo ">> Extracting generation operations..."
grep '\[gen\]' "$ORCH_LOG" > "$OUTPUT_DIR/gen_raw.log" 2>/dev/null || echo "No [gen] entries found"

# Extract rollout operations
echo ">> Extracting rollout operations..."
grep '\[rollout\]' "$ORCH_LOG" > "$OUTPUT_DIR/rollout_raw.log" 2>/dev/null || echo "No [rollout] entries found"

# Analyze queue_ms
echo ""
echo "===== Queue Delay Analysis ====="
if grep -q 'queue_ms=' "$OUTPUT_DIR/weights_raw.log" 2>/dev/null; then
    grep 'queue_ms=' "$OUTPUT_DIR/weights_raw.log" | \
        sed -n 's/.*queue_ms=\([0-9.]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            sumsq+=$1*$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            stddev = sqrt(sumsq/n - mean*mean);
            print "Count:", n;
            print "Mean:", mean, "ms";
            print "StdDev:", stddev, "ms";
            print "Min:", min, "ms";
            print "Max:", max, "ms";
            if (mean > 1000) {
                print "\n⚠️  WARNING: High queue delay detected!";
                print "   Average queue_ms > 1000ms indicates front-end saturation.";
            }
        }' | tee "$OUTPUT_DIR/queue_ms_stats.txt"
else
    echo "No queue_ms data found"
fi

# Analyze rpc_ms
echo ""
echo "===== RPC Time Analysis ====="
if grep -q 'rpc_ms=' "$OUTPUT_DIR/weights_raw.log" 2>/dev/null; then
    grep 'rpc_ms=' "$OUTPUT_DIR/weights_raw.log" | \
        sed -n 's/.*rpc_ms=\([0-9.]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            sumsq+=$1*$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            stddev = sqrt(sumsq/n - mean*mean);
            print "Count:", n;
            print "Mean:", mean, "ms";
            print "StdDev:", stddev, "ms";
            print "Min:", min, "ms";
            print "Max:", max, "ms";
            if (mean > 5000) {
                print "\n⚠️  WARNING: Slow weight loading detected!";
                print "   Average rpc_ms > 5000ms indicates slow weight loading.";
            }
        }' | tee "$OUTPUT_DIR/rpc_ms_stats.txt"
else
    echo "No rpc_ms data found"
fi

# Analyze checkpoint wait times
echo ""
echo "===== Checkpoint Wait Analysis ====="
if grep -q 'wait_ms=' "$OUTPUT_DIR/ckpt_raw.log" 2>/dev/null; then
    grep 'wait_ms=' "$OUTPUT_DIR/ckpt_raw.log" | \
        sed -n 's/.*wait_ms=\([0-9.]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            print "Count:", n;
            print "Mean:", mean, "ms";
            print "Min:", min, "ms";
            print "Max:", max, "ms";
            if (mean > 2000) {
                print "\n⚠️  WARNING: Slow checkpoint writes detected!";
                print "   Average wait_ms > 2000ms may indicate I/O bottleneck.";
            }
        }' | tee "$OUTPUT_DIR/ckpt_wait_stats.txt"
else
    echo "No checkpoint wait data found"
fi

# Analyze checkpoint write times
echo ""
echo "===== Checkpoint Write Analysis ====="
if grep -q 'write_ms=' "$OUTPUT_DIR/ckpt_raw.log" 2>/dev/null; then
    grep 'write_ms=' "$OUTPUT_DIR/ckpt_raw.log" | \
        sed -n 's/.*write_ms=\([0-9.]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            print "Count:", n;
            print "Mean:", mean, "ms";
            print "Min:", min, "ms";
            print "Max:", max, "ms";
        }' | tee "$OUTPUT_DIR/ckpt_write_stats.txt"
else
    echo "No checkpoint write data found"
fi

# Analyze generation batch times
echo ""
echo "===== Generation Batch Analysis ====="
if grep -q 'dur_ms=' "$OUTPUT_DIR/gen_raw.log" 2>/dev/null; then
    grep 'batch.done' "$OUTPUT_DIR/gen_raw.log" | \
        sed -n 's/.*dur_ms=\([0-9.]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            print "Count:", n;
            print "Mean:", mean, "ms";
            print "Min:", min, "ms";
            print "Max:", max, "ms";
        }' | tee "$OUTPUT_DIR/gen_batch_stats.txt"
else
    echo "No generation batch data found"
fi

# Analyze truncation
echo ""
echo "===== Truncation Analysis ====="
if grep -q 'trunc_pct=' "$OUTPUT_DIR/rollout_raw.log" 2>/dev/null; then
    grep 'trunc_pct=' "$OUTPUT_DIR/rollout_raw.log" | \
        sed -n 's/.*trunc_pct=\([0-9.]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            print "Count:", n;
            print "Mean:", mean, "%";
            print "Min:", min, "%";
            print "Max:", max, "%";
            if (mean > 20) {
                print "\n⚠️  WARNING: High truncation rate!";
                print "   Average truncation > 20% may affect training quality.";
            }
        }' | tee "$OUTPUT_DIR/truncation_stats.txt"
else
    echo "No truncation data found"
fi

# Analyze staleness
echo ""
echo "===== Staleness Analysis ====="
if grep -q 'staleness=' "$OUTPUT_DIR/rollout_raw.log" 2>/dev/null; then
    grep 'staleness=' "$OUTPUT_DIR/rollout_raw.log" | \
        sed -n 's/.*staleness=\([0-9]*\).*/\1/p' | \
        awk '{
            sum+=$1;
            n++;
            if ($1 > max) max=$1;
            if (min == "" || $1 < min) min=$1;
        } END {
            mean = sum/n;
            print "Count:", n;
            print "Mean:", mean, "steps";
            print "Min:", min, "steps";
            print "Max:", max, "steps";
        }' | tee "$OUTPUT_DIR/staleness_stats.txt"
else
    echo "No staleness data found"
fi

# Find slowest operations
echo ""
echo "===== Top 10 Slowest Weight Updates ====="
if grep -q 'wall_ms=' "$OUTPUT_DIR/weights_raw.log" 2>/dev/null; then
    grep 'client.done' "$OUTPUT_DIR/weights_raw.log" | \
        sort -t= -k2 -n | \
        tail -10 | \
        tee "$OUTPUT_DIR/slowest_updates.txt"
else
    echo "No wall_ms data found"
fi

# Correlate weight updates with generation
echo ""
echo "===== Timeline: Weight Updates vs Generation ====="
grep -E '\[gen\] batch\.(start|done)|\[weights\] client\.(send|done)' "$ORCH_LOG" 2>/dev/null | \
    tail -50 | \
    tee "$OUTPUT_DIR/timeline.txt" || echo "No timeline data found"

echo ""
echo "===== Analysis Complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
echo "--------"

# Generate summary
{
    echo "# Bottleneck Analysis Summary"
    echo "Generated: $(date)"
    echo ""

    if [ -f "$OUTPUT_DIR/queue_ms_stats.txt" ]; then
        echo "## Queue Delay"
        cat "$OUTPUT_DIR/queue_ms_stats.txt"
        echo ""
    fi

    if [ -f "$OUTPUT_DIR/rpc_ms_stats.txt" ]; then
        echo "## RPC Time"
        cat "$OUTPUT_DIR/rpc_ms_stats.txt"
        echo ""
    fi

    if [ -f "$OUTPUT_DIR/ckpt_wait_stats.txt" ]; then
        echo "## Checkpoint Wait"
        cat "$OUTPUT_DIR/ckpt_wait_stats.txt"
        echo ""
    fi

    echo "## Diagnosis"
    echo ""

    # Check for queue bottleneck
    if [ -f "$OUTPUT_DIR/queue_ms_stats.txt" ]; then
        MEAN_QUEUE=$(grep "^Mean:" "$OUTPUT_DIR/queue_ms_stats.txt" | awk '{print $2}')
        if (( $(echo "$MEAN_QUEUE > 1000" | bc -l) )); then
            echo "🔴 **PRIMARY BOTTLENECK: Front-end Queue Saturation**"
            echo "   - Average queue_ms = ${MEAN_QUEUE}ms"
            echo "   - Root cause: Weight update requests wait behind streaming inference"
            echo "   - Fix: Use dedicated admin client (already implemented)"
            echo ""
        fi
    fi

    # Check for RPC bottleneck
    if [ -f "$OUTPUT_DIR/rpc_ms_stats.txt" ]; then
        MEAN_RPC=$(grep "^Mean:" "$OUTPUT_DIR/rpc_ms_stats.txt" | awk '{print $2}')
        if (( $(echo "$MEAN_RPC > 5000" | bc -l) )); then
            echo "🔴 **SECONDARY BOTTLENECK: Slow Weight Loading**"
            echo "   - Average rpc_ms = ${MEAN_RPC}ms"
            echo "   - Root cause: Weight loading/swap operation is slow"
            echo "   - Consider: Faster storage, smaller model, or async loading"
            echo ""
        fi
    fi

} > "$OUTPUT_DIR/SUMMARY.md"

cat "$OUTPUT_DIR/SUMMARY.md"
