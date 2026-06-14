from __future__ import annotations

import itertools

import torch


def get_qwen3_5_vision_position_ids(
    *,
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    temp_merge_size: int = 1,
    time_interval: int = 1,
    device: torch.device | None = None,
) -> torch.LongTensor:
    llm_grid_t = int(grid_thw[0].item()) // temp_merge_size
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size

    position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
    position_height = torch.arange(llm_grid_h, device=device) + start_position
    position_width = torch.arange(llm_grid_w, device=device) + start_position

    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def build_qwen3_5_mrope_position_ids(
    *,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None,
    spatial_merge_size: int,
    seq_lens: torch.Tensor | None = None,
) -> torch.LongTensor:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape={tuple(input_ids.shape)}")
    if mm_token_type_ids.shape != input_ids.shape:
        raise ValueError(
            "mm_token_type_ids must match input_ids shape: "
            f"mm_token_type_ids={tuple(mm_token_type_ids.shape)}, input_ids={tuple(input_ids.shape)}"
        )
    if input_ids.shape[0] != 1:
        raise ValueError("Packed Qwen3.5 MRoPE builder currently supports batch size 1")

    total_seq_len = input_ids.shape[1]
    if seq_lens is None:
        seq_lens = torch.tensor([total_seq_len], dtype=torch.long, device=input_ids.device)
    else:
        seq_lens = seq_lens.to(device=input_ids.device, dtype=torch.long)

    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1D, got shape={tuple(seq_lens.shape)}")
    if bool((seq_lens <= 0).any().item()):
        raise ValueError(f"seq_lens must be positive, got {seq_lens.tolist()}")
    if int(seq_lens.sum().item()) != total_seq_len:
        raise ValueError(f"seq_lens sum must equal sequence length: {seq_lens.tolist()} vs {total_seq_len}")

    image_grids = iter(image_grid_thw) if image_grid_thw is not None else None
    all_positions = torch.empty(3, 1, total_seq_len, dtype=input_ids.dtype, device=input_ids.device)

    offset = 0
    for seq_len_tensor in seq_lens:
        seq_len = int(seq_len_tensor.item())
        token_types = mm_token_type_ids[0, offset : offset + seq_len]
        current_pos = 0
        segment_positions = []

        for modality_type, group in itertools.groupby(enumerate(token_types.tolist()), lambda item: item[1]):
            group = list(group)
            group_len = group[-1][0] - group[0][0] + 1

            if modality_type == 0:
                segment_positions.append(
                    torch.arange(group_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += group_len
                continue

            if modality_type == 2:
                raise ValueError("Qwen3.5 video MRoPE is not supported by the custom trainer path yet")
            if modality_type != 1:
                raise ValueError(f"Unsupported Qwen3.5 multimodal token type: {modality_type}")
            if image_grids is None:
                raise ValueError("image_grid_thw is required when mm_token_type_ids contains image tokens")

            try:
                grid_thw = next(image_grids)
            except StopIteration as exc:
                raise ValueError("Not enough image_grid_thw rows for image token groups") from exc

            vision_positions = get_qwen3_5_vision_position_ids(
                start_position=current_pos,
                grid_thw=grid_thw,
                spatial_merge_size=spatial_merge_size,
                device=input_ids.device,
            )
            if vision_positions.shape[1] != group_len:
                raise ValueError(
                    "Image token group length does not match image_grid_thw-derived token count: "
                    f"group_len={group_len}, grid_thw={grid_thw.tolist()}, "
                    f"derived={vision_positions.shape[1]}, spatial_merge_size={spatial_merge_size}"
                )
            segment_positions.append(vision_positions)
            current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size

        if not segment_positions:
            raise ValueError("Cannot build MRoPE positions for an empty sequence segment")

        segment_position_ids = torch.cat(segment_positions, dim=1)
        if segment_position_ids.shape[1] != seq_len:
            raise ValueError(
                f"Built MRoPE length does not match segment length: {segment_position_ids.shape[1]} vs {seq_len}"
            )
        all_positions[:, 0, offset : offset + seq_len] = segment_position_ids
        offset += seq_len

    if image_grids is not None:
        try:
            extra_grid = next(image_grids)
        except StopIteration:
            pass
        else:
            raise ValueError(f"Unused image_grid_thw row while building MRoPE positions: {extra_grid.tolist()}")

    return all_positions
