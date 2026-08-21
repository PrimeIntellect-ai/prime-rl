from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transports.rollouts import MicroBatch, MMImageRef, MMRefs


def test_wire_batch_keeps_compact_mm_refs_for_lazy_materialization():
    refs = MMRefs(images=[MMImageRef(url="data:image/png;base64,payload", offset=1, length=2)])
    wire_batch = MicroBatch(
        input_ids=[10, 11, 12],
        loss_mask=[False, True, True],
        advantages=[0.0, 1.0, 1.0],
        inference_logprobs=[0.0, -0.1, -0.2],
        position_ids=[0, 1, 2],
        sequence_lengths=[3],
        temperatures=[1.0, 1.0, 1.0],
        env_names=["test", "test", "test"],
        seq_lens=[3],
        mm_refs=refs,
        mm_token_type_ids=[0, 1, 1],
    )

    # Conversion must remain pure tensor/list conversion. In particular, it
    # must not invoke an image processor for every batch up front.
    loader = DataLoader.__new__(DataLoader)
    tensor_batch = loader._micro_batch_to_tensor(wire_batch)

    assert tensor_batch["mm_refs"] is refs
    assert tensor_batch["input_ids"].shape == (1, 3)
    assert tensor_batch["mm_token_type_ids"].shape == (1, 3)
