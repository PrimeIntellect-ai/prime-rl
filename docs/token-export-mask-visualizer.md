# Token Export Mask Visualizer

`scripts/token_export_mask_visualizer.py` renders a self-contained HTML page from prime-rl token export JSONL files. It is intended for debugging policy drift, token-level KL mismatch, and differences between DPPO and IcePop masking rules.

The visualizer expects token export files with fields such as `token_ids`, `loss_mask`, `mismatch_kl`, `prob_delta`, `importance_ratio`, `advantages`, `trainer_logprobs`, and `inference_logprobs`.

## Quick start

Run the script with `python` from the repository root. In a uv-managed environment, prefix the same commands with `uv run`.

Render the latest step in a run's `token_exports` directory:

```bash
python scripts/token_export_mask_visualizer.py /path/to/output/token_exports
```

Render a specific step and rank without loading a tokenizer:

```bash
python scripts/token_export_mask_visualizer.py /path/to/output/token_exports \
  --step 8 \
  --rank 0 \
  --no-decode
```

Render every matched rollout instead of only the most disagreement-heavy records:

```bash
python scripts/token_export_mask_visualizer.py /path/to/output/token_exports \
  --step 8 \
  --all-records \
  --no-decode
```

The generated HTML is self-contained and can be opened directly in a browser.

## Common options

- `--step N`: selects `step_N` under a `token_exports` root. Without this, the latest step directory is used.
- `--all-steps`: scans all step directories under the input.
- `--rank N`: filters to one trainer rank.
- `--env-name NAME`: filters to one environment name.
- `--top-records N`: embeds the top N records by mask disagreement. The default is `12`.
- `--all-records`: embeds every matched record.
- `--window-tokens N`: embeds a token window around the highest-disagreement or highest-KL token. The default is `2200`.
- `--full-record`: embeds complete selected records.
- `--tokenizer NAME_OR_PATH`: decodes token ids using a tokenizer.
- `--no-decode`: shows token ids only.
- `--allow-tokenizer-download`: permits downloading tokenizer files. By default, tokenizer loading uses local files only.

## Mask rules

The page compares two token-level masks:

- **DPPO**: uses advantage-sign-conditioned `prob_delta` thresholds.
- **IcePop**: uses lower and upper `importance_ratio` bounds.

DPPO defaults are read from `configs/trainer.toml` when that file is discoverable by walking up from the input path. If no config is found, DPPO thresholds default to `0.2` for both low and high sides. IcePop bounds default to `[0.2, 5.0]`.

Override thresholds from the CLI:

```bash
python scripts/token_export_mask_visualizer.py /path/to/output/token_exports \
  --dppo-mask-low 0.2 \
  --dppo-mask-high 0.2 \
  --icepop-ratio-low 0.2 \
  --icepop-ratio-high 5.0
```

The HTML UI can recompute both masks in the browser as these threshold controls are changed.

## Reading the page

- Red token background intensity shows `mismatch_kl`.
- Blue bottom rule marks tokens masked by DPPO.
- Orange top rule marks tokens masked by IcePop.
- When the KL gradient is disabled, purple fill marks tokens masked by both algorithms.
- Hover a token to inspect loss mask, mask reasons, KL, ratio, probabilities, `prob_delta`, and advantage.

The summary cards show global counts from the scanned data and live counts for the currently embedded token window.

## Agent workflow

When debugging a run, prefer this sequence:

1. Start with `--step`, `--rank`, and `--no-decode` for a fast smoke render.
2. Use the top-token table to identify whether disagreements are DPPO-only, IcePop-only, both, or high-KL unmasked tokens.
3. Re-run with `--all-records` when the initial page only embeds a few rollouts.
4. Add `--tokenizer` only when qualitative text inspection is needed and the tokenizer is locally available.
