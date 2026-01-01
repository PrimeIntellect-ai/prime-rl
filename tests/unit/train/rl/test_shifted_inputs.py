import torch


def _shift_inputs_right(labels: torch.Tensor) -> torch.Tensor:
    """Matches the RL trainer behavior: input_ids[:, 1:] = labels[:, :-1]."""
    x = torch.empty_like(labels)
    x[:, 0] = 0
    x[:, 1:] = labels[:, :-1]
    return x


def test_shift_then_shard_has_correct_left_context():
    # Simulate CP sharding: chunk along seq dim.
    labels = torch.arange(0, 16, dtype=torch.long).unsqueeze(0)  # [1, 16]
    cp_world_size = 4
    chunks = torch.chunk(labels, cp_world_size, dim=1)

    shifted_full = _shift_inputs_right(labels)
    shifted_chunks = torch.chunk(shifted_full, cp_world_size, dim=1)

    # For rank r>0, the first shifted token in that shard should equal the last label of rank r-1.
    for r in range(1, cp_world_size):
        assert shifted_chunks[r][0, 0].item() == chunks[r - 1][0, -1].item()
