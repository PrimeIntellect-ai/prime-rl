"""CPU checks for batch-to-trainer full-routing replay admission."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prime_rl.trainer.rl.data import DataLoader, prepare_router_replay
from prime_rl.trainer.routing_replay import RoutingReplay
from prime_rl.transports.batch.types import MicroBatch, RoutedExperts


def packed_batch(*, valid=(True, True, False), lengths=None, loss_mask=None, full=True):
    count = len(valid)
    ids = (np.arange(count * 4).reshape(count, 2, 2) % 4).astype(np.uint8)
    weights = np.broadcast_to(np.array([0.25, 0.75], dtype="<f4"), ids.shape).copy()
    weights[~np.asarray(valid)] = 0
    payload = RoutedExperts(data=ids.tobytes(), shape=list(ids.shape), dtype="uint8")
    if full:
        payload.weights = weights.tobytes()
        payload.valid = np.asarray(valid, dtype=np.uint8).tobytes()
        payload.format_version = 1
    lengths = lengths or [count]
    if loss_mask is None:
        loss_mask = [False] * count
        offset = 0
        for length in lengths:
            if length > 1:
                loss_mask[offset + length - 1] = True
            offset += length
    return (
        MicroBatch(
            input_ids=list(range(10, 10 + count)),
            loss_mask=loss_mask,
            advantages=[1.0] * count,
            inference_logprobs=[-1.0] * count,
            position_ids=[position for length in lengths for position in range(length)],
            sequence_lengths=list(lengths),
            temperatures=[1.0] * count,
            env_names=["test"] * count,
            seq_lens=list(lengths),
            routed_experts=payload,
        ),
        ids,
        weights,
    )


def decode(batch):
    # Exercise the real conversion method without opening a transport receiver.
    return DataLoader.__new__(DataLoader)._micro_batch_to_tensor(batch)


def model_config(**changes):
    fields = dict(model_type="qwen3_moe", num_hidden_layers=2, num_experts_per_tok=2, num_experts=4)
    return SimpleNamespace(**(fields | changes))


def prepare(batch, **changes):
    arguments = dict(enabled=True, mode="ids_and_weights", model_config=model_config()) | changes
    return prepare_router_replay(batch, **arguments)


def test_loader_materializes_owned_exact_pair_and_keeps_legacy_ids():
    batch, ids, weights = packed_batch()
    first = decode(batch)
    second = decode(batch)
    pair = first["routed_experts"]
    assert isinstance(pair, RoutingReplay)
    assert pair.ids.dtype == torch.int32 and pair.weights.dtype == torch.float32
    assert pair.ids.shape == pair.weights.shape == (1, 3, 2, 2)
    torch.testing.assert_close(pair.ids, torch.from_numpy(ids.astype(np.int32)).unsqueeze(0), rtol=0, atol=0)
    assert np.array_equal(pair.weights.numpy()[0].view(np.uint32), weights.view(np.uint32))
    pair.ids.zero_()
    pair.weights.zero_()
    assert np.array_equal(second["routed_experts"].weights.numpy()[0], weights)
    assert batch.routed_experts.weights == weights.tobytes()
    legacy, legacy_ids, _ = packed_batch(full=False)
    tensor = decode(legacy)["routed_experts"]
    assert isinstance(tensor, torch.Tensor)
    torch.testing.assert_close(tensor[0], torch.from_numpy(legacy_ids.astype(np.int32)), rtol=0, atol=0)


def test_loader_keeps_final_target_loss_when_only_its_input_row_is_uncaptured():
    batch, _, _ = packed_batch()
    tensor_batch = decode(batch)
    assert tensor_batch["loss_mask"][0, -1]
    assert isinstance(prepare(tensor_batch), RoutingReplay)


@pytest.mark.parametrize("terminal_valid", [False, True])
def test_loader_accepts_both_terminal_capture_states_followed_by_actual_packer_padding(terminal_valid):
    from prime_rl.trainer.batch import pad_micro_batch

    batch, _, _ = packed_batch(valid=(True, True, terminal_valid))
    padded = pad_micro_batch(batch, pad_to_multiple_of=8)
    pair = decode(padded)["routed_experts"]
    assert padded.sequence_lengths == padded.seq_lens == [8]
    assert pair.weights.shape == (1, 8, 2, 2)
    assert pair.weights[0, 3:].count_nonzero() == 0
    assert padded.routed_experts.valid[3:] == bytes(5)


def test_loader_accepts_terminal_placeholders_for_each_packed_document():
    batch, _, _ = packed_batch(valid=(True, True, False, True, True, False), lengths=[3, 3])
    pair = decode(batch)["routed_experts"]
    assert pair.weights[0, 2].count_nonzero() == pair.weights[0, 5].count_nonzero() == 0


@pytest.mark.parametrize("valid", [(False, True, False), (True, False, False)])
def test_loader_rejects_missing_causal_context_even_when_that_row_has_no_loss(valid):
    batch, _, _ = packed_batch(valid=valid, loss_mask=[False, False, True])
    with pytest.raises(ValueError, match="causal input row"):
        decode(batch)


def test_loader_allows_inactive_suffix_and_zero_loss_dummy_without_inventing_validity():
    batch, _, _ = packed_batch(valid=(True, False, False, False), loss_mask=[False, True, False, False])
    assert isinstance(decode(batch)["routed_experts"], RoutingReplay)
    dummy, _, _ = packed_batch(valid=(False, False, False), loss_mask=[False] * 3)
    pair = decode(dummy)["routed_experts"]
    assert pair.weights.count_nonzero() == 0


@pytest.mark.parametrize("stream", ["ce_weights", "ref_kl_weights"])
def test_loader_checks_non_rl_loss_streams_independent_of_loss_mask(stream):
    batch, _, _ = packed_batch(valid=(True, False, False), loss_mask=[False] * 3)
    setattr(batch, stream, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="causal input row"):
        decode(batch)


def test_loader_rejects_training_across_packed_document_boundary():
    batch, _, _ = packed_batch(valid=(True, True, False, True, True, False), lengths=[3, 3])
    batch.loss_mask[3] = True
    with pytest.raises(ValueError, match="loss-masked sequence starts"):
        decode(batch)


def test_loader_rejects_misaligned_attention_and_loss_boundaries():
    batch, _, _ = packed_batch()
    batch.seq_lens = [1, 2]
    with pytest.raises(ValueError, match="aligned positive sequence_lengths"):
        decode(batch)


def test_loader_rejects_nonzero_placeholder_weights():
    batch, _, weights = packed_batch()
    weights[-1] = 0.5
    batch.routed_experts.weights = weights.tobytes()
    with pytest.raises(ValueError, match="[Uu]ncaptured|[Ii]nvalid|placeholder"):
        decode(batch)


def test_loader_rejects_multimodal_full_replay():
    batch, _, _ = packed_batch()
    batch.mm_token_type_ids = [0, 0, 0]
    with pytest.raises(ValueError, match="multimodal"):
        decode(batch)


def test_prepare_selects_explicit_objective_without_implicit_fallback():
    batch, _, _ = packed_batch()
    tensor_batch = decode(batch)
    pair = tensor_batch["routed_experts"]
    assert prepare(tensor_batch) is pair
    assert prepare(tensor_batch, mode="ids") is pair.ids
    assert prepare(tensor_batch, enabled=False, mode="ids") is None
    with pytest.raises(ValueError, match="requires enable_router_replay"):
        prepare(tensor_batch, enabled=False)
    tensor_batch["routed_experts"] = pair.ids
    with pytest.raises(ValueError, match="IDs-only data is insufficient"):
        prepare(tensor_batch)
    assert prepare(tensor_batch, mode="ids") is pair.ids
    tensor_batch["routed_experts"] = None
    with pytest.raises(ValueError, match="requires routed experts"):
        prepare(tensor_batch)


@pytest.mark.parametrize("changes", [{"num_hidden_layers": 3}, {"num_experts_per_tok": 1}])
def test_prepare_checks_model_layer_and_topk_bounds_on_cpu(changes):
    batch, _, _ = packed_batch()
    with pytest.raises(ValueError, match="does not match expected"):
        prepare(decode(batch), model_config=model_config(**changes))


@pytest.mark.parametrize("bad_id", [-1, 4])
def test_prepare_checks_actual_model_expert_range_on_cpu(bad_id):
    batch, _, _ = packed_batch()
    tensor_batch = decode(batch)
    tensor_batch["routed_experts"].ids[0, 0, 0, 0] = bad_id
    with pytest.raises(ValueError, match="outside the model's expert range"):
        prepare(tensor_batch)


def test_prepare_rejects_mutated_nonfinite_coefficients_and_unsupported_model():
    batch, _, _ = packed_batch()
    tensor_batch = decode(batch)
    with pytest.raises(ValueError, match="only supports custom Qwen3"):
        prepare(tensor_batch, model_config=model_config(model_type="llama"))
    tensor_batch["routed_experts"].weights[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        prepare(tensor_batch)


@pytest.mark.parametrize("row_weights", [[0.0, 0.0], [0.2, 0.3], [0.8, 0.8]])
def test_loader_rejects_unnormalized_real_routing_rows(row_weights):
    batch, _, weights = packed_batch()
    weights[0, 0] = row_weights
    batch.routed_experts.weights = weights.tobytes()
    with pytest.raises(ValueError, match="normalized top-k"):
        decode(batch)


def test_loader_accepts_float32_normalization_roundoff_without_changing_values():
    batch, _, weights = packed_batch()
    weights[0, 0, 0] += np.spacing(np.float32(1.0))
    assert weights[0, 0].sum(dtype=np.float32) != 1.0
    batch.routed_experts.weights = weights.tobytes()
    pair = decode(batch)["routed_experts"]
    assert pair.weights.numpy()[0].tobytes() == weights.tobytes()


def test_loader_rejects_duplicate_real_experts_but_allows_terminal_placeholders():
    batch, ids, _ = packed_batch()
    ids[-1] = 0
    batch.routed_experts.data = ids.tobytes()
    decode(batch)  # Repeated dummy IDs are legal only on uncaptured zero-weight rows.
    ids[0, 0] = [1, 1]
    batch.routed_experts.data = ids.tobytes()
    with pytest.raises(ValueError, match="unique expert IDs"):
        decode(batch)


def test_loader_uses_actual_rl_membership_for_causal_validity():
    batch, _, _ = packed_batch(valid=(False, False, False), loss_mask=[True, True, True])
    batch.rl_weights = [0.0, 0.0, 0.0]
    assert isinstance(decode(batch)["routed_experts"], RoutingReplay)
    batch.rl_weights[-1] = 1.0
    with pytest.raises(ValueError, match="causal input row"):
        decode(batch)


def test_loader_rejects_misaligned_rl_weight_stream():
    batch, _, _ = packed_batch()
    batch.rl_weights = [0.0]
    with pytest.raises(ValueError, match="token-aligned rl_weights"):
        decode(batch)
