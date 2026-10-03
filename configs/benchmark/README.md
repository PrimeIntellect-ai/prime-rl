# Benchmark configs

Throughput benchmarks of the prime-rl SFT trainer against [torchtitan](https://github.com/pytorch/torchtitan),
with both stacks configured identically. Each config here has a torchtitan counterpart described below.
These runs measure step time only; they write no checkpoints.

## Qwen3-235B-A22B, 4 nodes, seq 16k

[`qwen3-235b-a22b/sft-4n-16k.toml`](qwen3-235b-a22b/sft-4n-16k.toml)

| | prime-rl | torchtitan |
|---|---|---|
| hardware | 4 nodes x 8 B300, FSDP over 32 GPUs, EP 8 | same |
| model | `Qwen/Qwen3-235B-A22B-Thinking-2507`, HF weights | Qwen3 235B-A22B, random init |
| tokens per step | 32 x 16384, one packed sample per GPU (`PrimeIntellect/Reverse-Text-SFT`) | 32 x 16384, one packed sample per GPU (`c4_test`) |
| routing | `debug.force_balanced_routing` (round-robin) | `debug.moe_force_load_balance` (round-robin) |
| optimizer | AdamW lr 1e-5, fp32 master weights and states, bf16 compute, fp32 reduce | `default_adamw`, same precisions |
| attention | FA4 varlen | FA4 varlen (`attn_backend="varlen"`) |
| activation checkpointing | full (recompute the block, keep `aten::topk`) | `TopKSaveAC` = selective AC saving only `aten.topk` |
| compile | `fullgraph=true` per block | `fullgraph=True` per block |
| steps | 20, steady state = steps 5 to 20 | same |

MFU uses the same formula in both stacks: `6 x active params (21.568B) + 6 x layers x heads x (qk_head_dim + v_head_dim) x seq`
with the real head dim of 128, i.e. 280.8 GFLOP/token at 16384, over a 2.25 PFLOP/s peak per B300. Note that this charges dense
causal attention while packed short documents make attention nearly free, so absolute MFU is roughly 2x overstated for both stacks;
the comparison is unaffected.

### Results (2026-10-02, back to back on the same nodes)

| run | s/step | tok/s/GPU | MFU | peak GiB |
|---|---|---|---|---|
| torchtitan `9e159aed7` | 5.04 | 3253 | 40.7% | 144 |
| prime-rl, this config | 4.39 | 3732 | 46.6% | 141 |

prime-rl's number requires #3753 (EP experts stay in the block's FSDP unit) and #3754 (FA4 custom op); without them the block cannot
compile with `fullgraph=true` and the same config runs at 7.0 s/step. The Mega MoE dispatch (#3651) brings the same setup to 4.10 s/step
with full AC and 3.93 s/step with selective AC; see `docs/performance.md` once it lands.

### Running it

```bash
uv run sft @ configs/benchmark/qwen3-235b-a22b/sft-4n-16k.toml
```

Real routing: drop `[model.debug]`. Measured at 6.11 s/step with the HF checkpoint's routing (prime-rl only; torchtitan has no
trained router to compare against).

### torchtitan counterpart

torchtitan `9e159aed7` with these additions to `torchtitan/models/qwen3/config_registry.py`, launched with
`torchrun --nnodes=4 --nproc-per-node=8 -m torchtitan.train --module qwen3 --config qwen3_235b_a22b_primerl_bench_4n_s16k_fullac_bal`
(torch 2.15 nightly, `PYTORCH_ALLOC_CONF=expandable_segments:True`):

```python
def qwen3_235b_a22b_primerl_bench_4n_s16k_fullac_bal(seq_len: int | None = 16384) -> Trainer.Config:
    model_config = model_registry("235B-A22B", seq_len=seq_len, attn_backend="varlen")
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_config))),
        hf_assets_path="./assets/hf/Qwen3-235B-A22B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]), shuffle=False),
        optimizer=default_adamw(lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=1),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=20,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1, expert_parallel_degree=8),
        compile=CompileConfig(components=["model", "loss"]),
        checkpointer=None,
        activation_checkpoint=TopKSaveAC.Config(),
    )
    config.debug = DebugConfig(moe_force_load_balance=True)
    return config


class TopKSaveAC(SelectiveAC):
    """Recompute the whole block except the router top-k: the counterpart of prime-rl's full AC.
    torchtitan's FullAC recomputes the router, and a re-routed token breaks the EP dispatch."""

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        force_recompute_mm_shapes_by_fqns: list[str] = field(default_factory=list)

    def get_save_ops(self) -> set:
        return {torch.ops.aten.topk.default}
```

torchtitan's selective AC OOMs at this size, and its full AC crashes under EP, hence `TopKSaveAC`.
