def test_packing_vs_padding():
    """
    Here we test that we don't lose any rewards or data when doing the different packing modes
    """

    BS = 32
    MICRO_BS = 4
    SEQ_LEN = 64

    batch_rollout = []

    seq_lens = [random.randint(1, SEQ_LEN) for _ in range(BS)]
    for i in range(BS):
        seq_len = seq_lens[i]
        data = {
            "input_ids": torch.ones(seq_len).int(),
            "advantages": torch.ones(seq_len),
            "loss_mask": torch.ones(seq_len).int(),
            "logprobs": torch.ones(seq_len),
            "seq_lens": torch.ones(seq_len),
            "rewards": torch.ones(1),
            "task_rewards": torch.ones(1),
            "length_penalties": torch.ones(1),
            "target_lengths": torch.ones(1),
            "task_type": "test_task",
        }

        batch_rollout.append(data)

    batch_packed = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, collate_mode="packing", micro_bs=MICRO_BS, pad_token_id=0)
    batch_padded = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, collate_mode="padding", micro_bs=MICRO_BS, pad_token_id=0)
    batch_balancing = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, collate_mode="balancing", micro_bs=MICRO_BS, pad_token_id=0)

    total_rewards_packed = sum(batch["rewards"].sum().item() for batch in batch_packed)
    total_rewards_padded = sum(batch["rewards"].sum().item() for batch in batch_padded)
    total_rewards_balancing = sum(batch["rewards"].sum().item() for batch in batch_balancing)

    assert total_rewards_padded == total_rewards_balancing
    assert total_rewards_packed == total_rewards_balancing

    total_input_ids_packed = sum(batch["input_ids"].sum().item() for batch in batch_packed)
    total_input_ids_padded = sum(batch["input_ids"].sum().item() for batch in batch_padded)
    total_input_ids_balancing = sum(batch["input_ids"].sum().item() for batch in batch_balancing)

    assert total_input_ids_packed == total_input_ids_padded
    assert total_input_ids_balancing == total_input_ids_padded

    total_padded_tokens_packed = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_packed) - total_input_ids_packed
    )
    total_padded_tokens_padded = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_padded) - total_input_ids_padded
    )
    total_padded_tokens_balancing = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_balancing) - total_input_ids_balancing
    )

    assert total_padded_tokens_packed < total_padded_tokens_padded
    assert total_padded_tokens_balancing <= total_padded_tokens_padded
