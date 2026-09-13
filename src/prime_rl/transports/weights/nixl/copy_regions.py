"""Propagate independent source regions through tensor layout operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import torch


@dataclass(frozen=True)
class CopyRegion:
    """A source value and the coordinates where it belongs in the output."""

    value: torch.Tensor
    offsets: tuple[int, ...]


def slice_regions(regions: list[CopyRegion], shape: tuple[int, ...], index) -> list[CopyRegion]:
    """Intersect source regions with basic indexing and update output coordinates."""
    indices = list(index if isinstance(index, tuple) else (index,))
    consumed_dimensions = sum(item is not None and item is not Ellipsis for item in indices)
    expanded = []
    for item in indices:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - consumed_dimensions))
        else:
            expanded.append(item)
    if not any(item is Ellipsis for item in indices):
        expanded.extend([slice(None)] * (len(shape) - consumed_dimensions))

    selected_regions = []
    for region in regions:
        source_index = []
        output_offsets = []
        dimension = 0
        for item in expanded:
            if item is None:
                source_index.append(None)
                output_offsets.append(0)
                continue
            input_start = region.offsets[dimension]
            input_end = input_start + region.value.shape[dimension]
            if isinstance(item, int):
                selected_index = item if item >= 0 else item + shape[dimension]
                if not input_start <= selected_index < input_end:
                    break
                source_index.append(selected_index - input_start)
            elif isinstance(item, slice):
                start, stop, step = item.indices(shape[dimension])
                first = max(0, (input_start - start + step - 1) // step)
                last = min(len(range(start, stop, step)), (input_end - start + step - 1) // step)
                if first >= last:
                    break
                local_start = start + first * step - input_start
                local_stop = min(input_end - input_start, local_start + (last - first) * step)
                source_index.append(slice(local_start, local_stop, step))
                output_offsets.append(first)
            else:
                raise NotImplementedError(f"unsupported region index {item!r}")
            dimension += 1
        else:
            selected_regions.append(CopyRegion(region.value[tuple(source_index)], tuple(output_offsets)))
    return selected_regions


def permute_regions(regions: list[CopyRegion], dimensions: tuple[int, ...]) -> list[CopyRegion]:
    """Permute source values and their output coordinates together."""
    return [
        CopyRegion(region.value.permute(dimensions), tuple(region.offsets[dim] for dim in dimensions))
        for region in regions
    ]


def split_flat_interval(start: int, length: int, shape: tuple[int, ...]):
    """Partition a flat interval into contiguous rectangular output regions."""
    if not shape:
        yield (), (), 1
        return
    strides = tuple(prod(shape[dim + 1 :]) for dim in range(len(shape)))
    while length:
        coordinates = tuple(start // stride % size for size, stride in zip(shape, strides, strict=True))
        for dimension, stride in enumerate(strides):
            if start % stride == 0 and length >= stride:
                width = min(shape[dimension] - coordinates[dimension], length // stride)
                region_shape = (1,) * dimension + (width,) + shape[dimension + 1 :]
                region_length = prod(region_shape)
                yield coordinates, region_shape, region_length
                start += region_length
                length -= region_length
                break


def reshape_regions(
    regions: list[CopyRegion], old_shape: tuple[int, ...], new_shape: tuple[int, ...]
) -> list[CopyRegion]:
    """Reshape logical regions, splitting only where output row boundaries require it."""
    if prod(new_shape) == 0:
        return []
    if old_shape == new_shape:
        return regions
    prefix_dimensions = 0
    for old_size, new_size in zip(old_shape, new_shape):
        if old_size != new_size:
            break
        prefix_dimensions += 1
    old_suffix = old_shape[prefix_dimensions:]
    new_suffix = new_shape[prefix_dimensions:]
    old_strides = tuple(prod(old_suffix[dim + 1 :]) for dim in range(len(old_suffix)))
    reshaped_regions = []
    for region in regions:
        prefix_shape = tuple(region.value.shape[:prefix_dimensions])
        prefix_offsets = region.offsets[:prefix_dimensions]
        region_shape = tuple(region.value.shape[prefix_dimensions:])
        region_offsets = region.offsets[prefix_dimensions:]
        split_dimension = len(old_suffix) - 1
        while split_dimension > 0 and region_shape[split_dimension] == old_suffix[split_dimension]:
            split_dimension -= 1
        run_length = prod(region_shape[split_dimension:])
        flattened_source = region.value.reshape(*prefix_shape, prod(region_shape))
        source_offset = 0
        for outer_index in product(*(range(size) for size in region_shape[:split_dimension])):
            coordinates = list(region_offsets)
            for dimension, index in enumerate(outer_index):
                coordinates[dimension] += index
            flat_start = sum(index * stride for index, stride in zip(coordinates, old_strides, strict=True))
            for output_offsets, output_shape, length in split_flat_interval(flat_start, run_length, new_suffix):
                source_slice = flattened_source.narrow(prefix_dimensions, source_offset, length)
                source_value = source_slice.reshape(prefix_shape + output_shape)
                reshaped_regions.append(CopyRegion(source_value, prefix_offsets + output_offsets))
                source_offset += length
    return reshaped_regions


def transform_regions(regions: list[CopyRegion], meta: torch.Tensor, operation, output_index: int | None = None):
    """Apply a recorded operation to the regions and the complete logical shape."""
    name, args, kwargs = operation.name, operation.args, operation.kwargs
    shape = tuple(meta.shape)
    result = getattr(meta, name)(*args, **kwargs)
    if name == "__getitem__":
        return slice_regions(regions, shape, args[0]), result
    if name in ("narrow", "select"):
        dimension = kwargs.get("dim", args[0] if args else None) % meta.ndim
        start = kwargs.get("start" if name == "narrow" else "index", args[1] if len(args) > 1 else None)
        start = start if start >= 0 else start + shape[dimension]
        index = [slice(None)] * meta.ndim
        if name == "narrow":
            length = kwargs.get("length", args[2] if len(args) > 2 else None)
            index[dimension] = slice(start, start + length)
        else:
            index[dimension] = start
        return slice_regions(regions, shape, tuple(index)), result
    if name in ("chunk", "split", "unbind"):
        argument_position = 0 if name == "unbind" else 1
        dimension = kwargs.get("dim", args[argument_position] if len(args) > argument_position else 0) % meta.ndim
        assert output_index is not None
        output_index %= len(result)
        index = [slice(None)] * meta.ndim
        if name == "unbind":
            index[dimension] = output_index
        else:
            start = sum(output.shape[dimension] for output in result[:output_index])
            index[dimension] = slice(start, start + result[output_index].shape[dimension])
        return slice_regions(regions, shape, tuple(index)), result[output_index]
    if name in ("transpose", "t", "permute"):
        dimensions = list(range(meta.ndim))
        if name == "transpose":
            first = kwargs.get("dim0", args[0] if args else None)
            second = kwargs.get("dim1", args[1] if len(args) > 1 else None)
            dimensions[first], dimensions[second] = dimensions[second], dimensions[first]
        elif name == "t":
            dimensions.reverse()
        else:
            dimensions = kwargs.get("dims", args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args)
        return permute_regions(regions, tuple(dimensions)), result
    if name in ("view", "reshape", "flatten", "unsqueeze", "squeeze"):
        return reshape_regions(regions, shape, tuple(result.shape)), result
    if name in ("contiguous", "to", "float", "bfloat16"):
        return [CopyRegion(getattr(region.value, name)(*args, **kwargs), region.offsets) for region in regions], result
    raise NotImplementedError(f"unsupported region operation {name!r}")
