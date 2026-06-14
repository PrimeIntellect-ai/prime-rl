from collections.abc import Sequence

import numpy as np

from prime_rl.transport.types import PackedArray, TrainingSample


def _pack_numeric(values: Sequence[int] | Sequence[float], dtype: str) -> PackedArray:
    arr = np.asarray(values, dtype=np.dtype(dtype))
    return PackedArray(data=arr.tobytes(), shape=[int(arr.shape[0])], dtype=dtype)


def _pack_bool(values: Sequence[bool]) -> PackedArray:
    arr = np.asarray(values, dtype=np.bool_)
    return PackedArray(data=np.packbits(arr, bitorder="little").tobytes(), shape=[int(arr.shape[0])], dtype="bool")


def _unpack_numeric(packed: PackedArray) -> list[int] | list[float]:
    return np.frombuffer(packed.data, dtype=np.dtype(packed.dtype), count=packed.shape[0]).tolist()


def _unpack_bool(packed: PackedArray) -> list[bool]:
    if packed.shape[0] == 0:
        return []
    packed_bytes = np.frombuffer(packed.data, dtype=np.uint8)
    return np.unpackbits(packed_bytes, bitorder="little", count=packed.shape[0]).astype(np.bool_).tolist()


def _packed_len(packed: PackedArray | None) -> int | None:
    if packed is None:
        return None
    return packed.shape[0]


def _field_len(values: Sequence | None, packed: PackedArray | None) -> int:
    packed_len = _packed_len(packed)
    return packed_len if packed_len is not None else len(values or [])


def compact_training_sample(sample: TrainingSample) -> None:
    """Replace large list fields with byte-backed arrays for transport."""
    if sample.packed_prompt_ids is not None:
        return

    sample.packed_prompt_ids = _pack_numeric(sample.prompt_ids, "uint32")
    sample.prompt_ids = []
    sample.packed_prompt_mask = _pack_bool(sample.prompt_mask)
    sample.prompt_mask = []
    sample.packed_completion_ids = _pack_numeric(sample.completion_ids, "uint32")
    sample.completion_ids = []
    sample.packed_completion_mask = _pack_bool(sample.completion_mask)
    sample.completion_mask = []
    sample.packed_completion_logprobs = _pack_numeric(sample.completion_logprobs, "float32")
    sample.completion_logprobs = []

    if sample.completion_temperatures:
        sample.packed_completion_temperatures = _pack_numeric(sample.completion_temperatures, "float32")
        sample.completion_temperatures = []

    if sample.teacher_logprobs is not None:
        sample.packed_teacher_logprobs = _pack_numeric(sample.teacher_logprobs, "float32")
        sample.teacher_logprobs = None

    if sample.mm_token_type_ids is not None:
        sample.packed_mm_token_type_ids = _pack_numeric(sample.mm_token_type_ids, "uint8")
        sample.mm_token_type_ids = None


def compact_training_samples(samples: list[TrainingSample]) -> None:
    for sample in samples:
        compact_training_sample(sample)


def training_sample_prompt_ids(sample: TrainingSample) -> list[int]:
    if sample.packed_prompt_ids is not None:
        return _unpack_numeric(sample.packed_prompt_ids)
    return sample.prompt_ids


def training_sample_prompt_mask(sample: TrainingSample) -> list[bool]:
    if sample.packed_prompt_mask is not None:
        return _unpack_bool(sample.packed_prompt_mask)
    return sample.prompt_mask


def training_sample_completion_ids(sample: TrainingSample) -> list[int]:
    if sample.packed_completion_ids is not None:
        return _unpack_numeric(sample.packed_completion_ids)
    return sample.completion_ids


def training_sample_completion_mask(sample: TrainingSample) -> list[bool]:
    if sample.packed_completion_mask is not None:
        return _unpack_bool(sample.packed_completion_mask)
    return sample.completion_mask


def training_sample_completion_logprobs(sample: TrainingSample) -> list[float]:
    if sample.packed_completion_logprobs is not None:
        return _unpack_numeric(sample.packed_completion_logprobs)
    return sample.completion_logprobs


def training_sample_completion_temperatures(sample: TrainingSample) -> list[float]:
    if sample.packed_completion_temperatures is not None:
        return _unpack_numeric(sample.packed_completion_temperatures)
    return sample.completion_temperatures


def training_sample_teacher_logprobs(sample: TrainingSample) -> list[float] | None:
    if sample.packed_teacher_logprobs is not None:
        return _unpack_numeric(sample.packed_teacher_logprobs)
    return sample.teacher_logprobs


def training_sample_mm_token_type_ids(sample: TrainingSample) -> list[int] | None:
    if sample.packed_mm_token_type_ids is not None:
        return _unpack_numeric(sample.packed_mm_token_type_ids)
    return sample.mm_token_type_ids


def training_sample_prompt_len(sample: TrainingSample) -> int:
    return _field_len(sample.prompt_ids, sample.packed_prompt_ids)


def training_sample_prompt_mask_len(sample: TrainingSample) -> int:
    return _field_len(sample.prompt_mask, sample.packed_prompt_mask)


def training_sample_completion_len(sample: TrainingSample) -> int:
    return _field_len(sample.completion_ids, sample.packed_completion_ids)


def training_sample_completion_mask_len(sample: TrainingSample) -> int:
    return _field_len(sample.completion_mask, sample.packed_completion_mask)


def training_sample_completion_logprobs_len(sample: TrainingSample) -> int:
    return _field_len(sample.completion_logprobs, sample.packed_completion_logprobs)


def training_sample_completion_temperatures_len(sample: TrainingSample) -> int:
    return _field_len(sample.completion_temperatures, sample.packed_completion_temperatures)


def training_sample_teacher_logprobs_len(sample: TrainingSample) -> int | None:
    packed_len = _packed_len(sample.packed_teacher_logprobs)
    if packed_len is not None:
        return packed_len
    if sample.teacher_logprobs is None:
        return None
    return len(sample.teacher_logprobs)


def training_sample_token_len(sample: TrainingSample) -> int:
    return training_sample_prompt_len(sample) + training_sample_completion_len(sample)
