"""Split recorded concatenations into independent source-to-destination copies."""

from dataclasses import replace
from itertools import product
from math import prod

import torch

from prime_rl.transports.weights.nixl.graph import RecordedCopy, TensorOperation, apply_chain


def destination_coordinates(copy: RecordedCopy, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Locate a copy in a contiguous logical output before binding physical strides."""
    strides = torch.empty(shape, device="meta").stride()
    return tuple(copy.destination_offset // stride % size for size, stride in zip(shape, strides, strict=True))


def place_copy(
    copy: RecordedCopy, coordinates: tuple[int, ...], shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> RecordedCopy:
    """Position a copy within the current logical output."""
    strides = torch.empty(output_shape, device="meta").stride()
    return replace(
        copy,
        destination_offset=sum(offset * stride for offset, stride in zip(coordinates, strides, strict=True)),
        destination_shape=shape,
        destination_stride=strides,
    )


def slice_copies(
    copies: list[RecordedCopy], shape: tuple[int, ...], index, output_shape: tuple[int, ...]
) -> list[RecordedCopy]:
    """Intersect a slice with each copy and translate it into source-local indexing."""
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

    selected_copies = []
    for copy in copies:
        input_offsets = destination_coordinates(copy, shape)
        source_index = []
        output_offsets = []
        dimension = 0
        for item in expanded:
            if item is None:
                source_index.append(None)
                output_offsets.append(0)
                continue
            input_start = input_offsets[dimension]
            input_end = input_start + copy.destination_shape[dimension]
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
                raise NotImplementedError(f"unsupported copy index {item!r}")
            dimension += 1
        else:
            operation = TensorOperation("__getitem__", args=(tuple(source_index),))
            source_shape = apply_chain(torch.empty(copy.destination_shape, device="meta"), (operation,)).shape
            selected_copies.append(
                place_copy(
                    replace(copy, ops=copy.ops + (operation,)), tuple(output_offsets), source_shape, output_shape
                )
            )
    return selected_copies


def split_flat_interval(start: int, length: int, shape: tuple[int, ...]):
    """Partition a flat interval into contiguous rectangular output slices."""
    if not shape:
        yield (), (), 1
        return
    strides = tuple(prod(shape[dim + 1 :]) for dim in range(len(shape)))
    while length:
        coordinates = tuple(start // stride % size for size, stride in zip(shape, strides, strict=True))
        for dimension, stride in enumerate(strides):
            if start % stride == 0 and length >= stride:
                width = min(shape[dimension] - coordinates[dimension], length // stride)
                copy_shape = (1,) * dimension + (width,) + shape[dimension + 1 :]
                copy_length = prod(copy_shape)
                yield coordinates, copy_shape, copy_length
                start += copy_length
                length -= copy_length
                break


def reshape_copies(
    copies: list[RecordedCopy], old_shape: tuple[int, ...], new_shape: tuple[int, ...]
) -> list[RecordedCopy]:
    """Split copies where reshaping makes their logical destination slices nonrectangular."""
    if prod(new_shape) == 0:
        return []
    if old_shape == new_shape:
        return copies
    prefix_dimensions = 0
    for old_size, new_size in zip(old_shape, new_shape):
        if old_size != new_size:
            break
        prefix_dimensions += 1
    old_suffix = old_shape[prefix_dimensions:]
    new_suffix = new_shape[prefix_dimensions:]
    old_strides = tuple(prod(old_suffix[dim + 1 :]) for dim in range(len(old_suffix)))
    reshaped_copies = []
    for copy in copies:
        coordinates = destination_coordinates(copy, old_shape)
        prefix_shape = copy.destination_shape[:prefix_dimensions]
        prefix_offsets = coordinates[:prefix_dimensions]
        copy_shape = copy.destination_shape[prefix_dimensions:]
        copy_offsets = coordinates[prefix_dimensions:]
        split_dimension = len(old_suffix) - 1
        while split_dimension > 0 and copy_shape[split_dimension] == old_suffix[split_dimension]:
            split_dimension -= 1
        run_length = prod(copy_shape[split_dimension:])
        flatten_source = TensorOperation("reshape", args=(prefix_shape + (prod(copy_shape),),))
        source_offset = 0
        for outer_index in product(*(range(size) for size in copy_shape[:split_dimension])):
            coordinates = list(copy_offsets)
            for dimension, index in enumerate(outer_index):
                coordinates[dimension] += index
            flat_start = sum(index * stride for index, stride in zip(coordinates, old_strides, strict=True))
            for output_offsets, output_shape, length in split_flat_interval(flat_start, run_length, new_suffix):
                source_shape = prefix_shape + output_shape
                ops = copy.ops + (
                    flatten_source,
                    TensorOperation("narrow", args=(prefix_dimensions, source_offset, length)),
                    TensorOperation("reshape", args=(source_shape,)),
                )
                reshaped_copies.append(
                    place_copy(replace(copy, ops=ops), prefix_offsets + output_offsets, source_shape, new_shape)
                )
                source_offset += length
    return reshaped_copies


def transform_copies(
    copies: list[RecordedCopy], meta: torch.Tensor, operation: TensorOperation, output_index: int | None = None
) -> tuple[list[RecordedCopy], torch.Tensor]:
    """Translate an operation after cat into each branch's source chain and destination layout."""
    name, args, kwargs = operation.name, operation.args, operation.kwargs
    shape = tuple(meta.shape)
    result = getattr(meta, name)(*args, **kwargs)
    if name == "__getitem__":
        return slice_copies(copies, shape, args[0], result.shape), result
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
        return slice_copies(copies, shape, tuple(index), result.shape), result
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
        return slice_copies(copies, shape, tuple(index), result[output_index].shape), result[output_index]
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
        operation = TensorOperation("permute", args=(tuple(dimensions),))
        permuted_copies = []
        for copy in copies:
            offsets = destination_coordinates(copy, shape)
            permuted_copies.append(
                place_copy(
                    replace(copy, ops=copy.ops + (operation,)),
                    tuple(offsets[dim] for dim in dimensions),
                    tuple(copy.destination_shape[dim] for dim in dimensions),
                    result.shape,
                )
            )
        return permuted_copies, result
    if name in ("view", "reshape", "flatten", "unsqueeze", "squeeze"):
        return reshape_copies(copies, shape, tuple(result.shape)), result
    if name in ("contiguous", "to", "float", "bfloat16"):
        return [replace(copy, ops=copy.ops + (operation,)) for copy in copies], result
    raise NotImplementedError(f"unsupported copy operation {name!r}")


def split_concatenated_copy(copy: RecordedCopy) -> list[RecordedCopy]:
    """Branch at cat operations and return copies containing only ordinary source chains."""
    cat_index = next((index for index, operation in enumerate(copy.ops) if operation.name == "cat"), None)
    if cat_index is None:
        return [copy]

    concatenation = copy.ops[cat_index]
    (inputs,) = concatenation.args
    meta = apply_chain(torch.empty((), device="meta"), (concatenation,))
    dimension = concatenation.kwargs["dim"] % meta.ndim
    copies = []
    concat_offset = 0
    for weight in inputs:
        if weight.numel() == 0:
            continue
        input_copy = replace(copy, source_name=weight._source_name, ops=weight._ops)
        input_copy = place_copy(input_copy, (0,) * weight.ndim, tuple(weight.shape), tuple(weight.shape))
        for child_copy in split_concatenated_copy(input_copy):
            offsets = list(destination_coordinates(child_copy, tuple(weight.shape)))
            offsets[dimension] += concat_offset
            if weight.dtype != meta.dtype:
                child_copy = replace(
                    child_copy, ops=child_copy.ops + (TensorOperation("to", kwargs={"dtype": meta.dtype}),)
                )
            copies.append(place_copy(child_copy, tuple(offsets), child_copy.destination_shape, tuple(meta.shape)))
        concat_offset += weight.shape[dimension]

    operations = iter(copy.ops[cat_index + 1 :])
    for operation in operations:
        output_index = None
        if operation.name in ("chunk", "split", "unbind"):
            selection = next(operations)
            assert selection.name == "tuple_getitem"
            output_index = selection.args[0]
        copies, meta = transform_copies(copies, meta, operation, output_index)

    result = []
    for child_copy in copies:
        offsets = destination_coordinates(child_copy, tuple(meta.shape))
        result.append(
            replace(
                child_copy,
                destination_offset=copy.destination_offset
                + sum(offset * stride for offset, stride in zip(offsets, copy.destination_stride, strict=True)),
                destination_stride=copy.destination_stride,
            )
        )
    return result
