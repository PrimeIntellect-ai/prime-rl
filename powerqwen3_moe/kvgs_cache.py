import copy
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)

_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys, self.values, self.gatings, self.state, self.sum_of_keys = None, None, None, None, None

    @abstractmethod
    def update_kv(
        self, key_states: torch.Tensor, value_states: torch.Tensor, gate_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def update_state(
        self, state: torch.Tensor, sum_of_keys: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def lazy_initialization_state(self, state: torch.Tensor): ...

    @abstractmethod
    def get_seq_length(self, cache_position=None) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.keys is not None:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)
            self.gatings = self.gatings.to("cpu", non_blocking=True)
            self.state = self.state.to("cpu", non_blocking=True)
            self.sum_of_keys = self.sum_of_keys.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.keys is not None and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)
            self.gatings = self.gatings.to(self.device, non_blocking=True)
            self.state = self.state.to(self.device, non_blocking=True)
            self.sum_of_keys = self.sum_of_keys.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()
            self.gatings.zero_()
            self.state.zero_()
            self.sum_of_keys.zero_()

    def clean_kv(self) -> None:
        if self.keys is not None:
            self.keys = None
            self.values = None
            self.gatings = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reorders this layer's cache for beam search."""
        if self.keys.numel():
            device = self.keys.device
            self.keys = self.keys.index_select(0, beam_idx.to(device))
        if self.values.numel():
            device = self.values.device
            self.values = self.values.index_select(0, beam_idx.to(device))
        if self.gatings.numel():
            device = self.gatings.device
            self.gatings = self.gatings.index_select(0, beam_idx.to(device))
        if self.state.numel():
            device = self.state.device
            self.state = self.state.index_select(0, beam_idx.to(device))
        if self.sum_of_keys.numel():
            device = self.sum_of_keys.device
            self.sum_of_keys = self.sum_of_keys.index_select(0, beam_idx.to(device))


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the Key and Value states as tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.gatings = torch.tensor([], dtype=torch.float32, device=self.device)

    def lazy_initialization_state(self, state: torch.Tensor):
        self.state = torch.tensor([], dtype=torch.float32, device=self.device)
        self.sum_of_keys = torch.tensor([], dtype=torch.float32, device=self.device)

    def update_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        gate_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            gate_states (`torch.Tensor`):
                The new gate states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicLayer`.

        Return:
            A tuple containing the updated key and value states, and current state and sum of keys.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        self.gatings = torch.cat([self.gatings, gate_states], dim=-1)
        return self.keys, self.values, self.gatings, self.state, self.sum_of_keys

    def update_state(
        self, state: torch.Tensor, sum_of_keys: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization
        if self.state is None:
            self.lazy_initialization_state(state)

        self.state = state
        self.sum_of_keys = sum_of_keys
        return self.state, self.sum_of_keys

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders the cache for beam search, given the selected beam indices."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))
            self.gatings = self.gatings.index_select(0, beam_idx.to(self.gatings.device))
            self.state = self.state.index_select(0, beam_idx.to(self.state.device))
            self.sum_of_keys = self.sum_of_keys.index_select(0, beam_idx.to(self.sum_of_keys.device))

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        if self.keys is not None and self.keys.numel():
            self.keys = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]
            self.gatings = self.gatings[..., :max_length]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
            self.gatings = self.gatings.repeat_interleave(repeats, dim=0)
            self.state = self.state.repeat_interleave(repeats, dim=0)
            self.sum_of_keys = self.sum_of_keys.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
            self.gatings = self.gatings[indices, ...]
            self.state = self.state[indices, ...]
            self.sum_of_keys = self.sum_of_keys[indices, ...]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @classmethod
    def from_tensors(cls, keys: torch.Tensor, values: torch.Tensor, gatings: torch.Tensor, state: torch.Tensor, sum_of_keys: torch.Tensor) -> "DynamicLayer":
        """
        Build a `DynamicLayer` instance from pre-existing key/value tensors.

        Args:
            keys (`torch.Tensor`):
                Key cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
            values (`torch.Tensor`):
                Value cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
            gatings (`torch.Tensor`):
                Gating cache tensor of shape ``[batch_size, num_heads, seq_len]``.

        Returns:
            `DynamicLayer`: The newly constructed layer whose internal cache directly references
            the supplied tensors.
        """
        layer = cls()
        layer.dtype, layer.device = keys.dtype, keys.device
        layer.keys = keys
        layer.values = values
        layer.gatings = gatings
        layer.state = state
        layer.sum_of_keys = sum_of_keys
        return layer


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the Key and Value states as static tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.
    It allocates its full backing tensors up-front and mutates them in-place. Built for `torch.compile` support.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
        """
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.gatings = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len),
            dtype=torch.float32,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.gatings)


    def lazy_initialization_state(self, state: torch.Tensor):
        self.state = torch.zeros(
            (self.max_batch_size, self.num_heads, self.D, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.sum_of_keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len),
            dtype=torch.float32,
            device=self.device,
        )
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.state)
            torch._dynamo.mark_static_address(self.sum_of_keys)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        gate_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update the static cache tensors in place.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            gate_states (`torch.Tensor`): The new gate states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`, `torch.Tensor`, `torch.Tensor`]: The updated key, value, and gate states, and current state and sum of keys.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
            self.gatings.index_copy_(2, cache_position, gate_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
            self.gatings[:, :, cache_position] = gate_states
        return self.keys, self.values, self.gatings, self.state, self.sum_of_keys

    def update_state(
        self, state: torch.Tensor, sum_of_keys: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization
        if self.state is None:
            self.lazy_initialization_state(state)

        self.state = state
        self.sum_of_keys = sum_of_keys
        return self.state, self.sum_of_keys

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if cache_position is not None:
            return int(cache_position[-1] + 1)
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        seq_length = (self.keys[0, 0].any(dim=-1)).sum() if self.keys is not None else 0
        return seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders the cache for beam search, given the selected beam indices."""
        dev = self.keys.device
        beam_idx_dev = beam_idx.to(dev)
        self.keys = self.keys.index_select(0, beam_idx_dev)
        self.values = self.values.index_select(0, beam_idx_dev)
        self.gatings = self.gatings.index_select(0, beam_idx_dev)
        self.state = self.state.index_select(0, beam_idx_dev)
        self.sum_of_keys = self.sum_of_keys.index_select(0, beam_idx_dev)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset



class KeyValuesGatingStateWrapper:
    """Helper class for Cache that simulates layer-indexed key/value lists from a layered cache.
    This allows for BC access and writing, e.g., cache.key_cache[idx] = ...
    Deprecated in favor of Cache.layers[idx].keys/values. TODO: remove in v4.56.0"""

    def __init__(self, layers, cache_type="keys"):
        self.layers = layers
        self.cache_type = cache_type

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [getattr(layer, self.cache_type) for layer in self.layers[idx]]
        return getattr(self.layers[idx], self.cache_type)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            for layer, val in zip(self.layers[idx], value):
                setattr(layer, self.cache_type, val)
        else:
            setattr(self.layers[idx], self.cache_type, value)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        for layer in self.layers:
            yield getattr(layer, self.cache_type)

    def __bool__(self):
        return bool(self.layers)


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Parameters:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        if only_non_sliding:
            # Try to find next non-sliding, starting at `layer_idx`
            try:
                layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
            # In this case, we need to circle back to the begining
            except ValueError:
                layer_idx = self.is_sliding.index(False)
        else:
            layer_idx = layer_idx if layer_idx < len(self.layers) else 0

        # Prefetch
        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            self.layers[layer_idx].prefetch()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if not (only_non_sliding and self.is_sliding[layer_idx]):
            self.layers[layer_idx].offload()

    def update_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        gate_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            gate_states (`torch.Tensor`):
                The new gate states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key, value, and gate states, and current state and sum of keys.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values, gatings, state, sum_of_keys = self.layers[layer_idx].update_kv(key_states, value_states, gate_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values, gatings, state, sum_of_keys

    def clean_kv(self, layer_idx: int) -> None:
        self.layers[layer_idx].clean_kv()

    def update_state(
        self, state: torch.Tensor, sum_of_keys: torch.Tensor, layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        
        state, sum_of_keys = self.layers[layer_idx].update_state(state, sum_of_keys, cache_kwargs)

        return state, sum_of_keys

    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, D: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        fake_state_tensor = torch.zeros((batch_size, num_heads, D, head_dim), dtype=dtype, device=device)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)
            layer.lazy_initialization_state(fake_state_tensor)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length(cache_position)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values, self.layers[layer_idx].gatings, self.layers[layer_idx].state, self.layers[layer_idx].sum_of_keys
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values, self.layers[layer_idx].gatings)

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)

    @property
    def key_cache(self) -> KeyValuesGatingStateWrapper:
        """List-like object of key cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].keys`"""
        logger.warning_once(
            "`cache.key_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].keys` instead."
        )
        return KeyValuesGatingStateWrapper(self.layers, "keys")

    @property
    def value_cache(self) -> KeyValuesGatingStateWrapper:
        """List-like object of value cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].values`"""
        logger.warning_once(
            "`cache.value_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].values` instead."
        )
        return KeyValuesGatingStateWrapper(self.layers, "values")

    @property
    def gating_cache(self) -> KeyValuesGatingStateWrapper:
        """List-like object of gate cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].gatings`"""
        logger.warning_once(
            "`cache.gate_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].gatings` instead."
        )
        return KeyValuesGatingStateWrapper(self.layers, "gatings")

    @property
    def state_cache(self) -> KeyValuesGatingStateWrapper:
        """List-like object of state cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].state`"""
        logger.warning_once(
            "`cache.state_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].state` instead."
        )
        return KeyValuesGatingStateWrapper(self.layers, "state")
    
    @property
    def sum_of_keys_cache(self) -> KeyValuesGatingStateWrapper:
        """List-like object of sum of keys cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].sum_of_keys`"""
        logger.warning_once(
            "`cache.sum_of_keys_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].sum_of_keys` instead."
        )
        return KeyValuesGatingStateWrapper(self.layers, "sum_of_keys")

class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key, Value, and Gating states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]` for Key and Value, and `[batch_size, num_heads, seq_len]` for Gating.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    # Specialized constructor for DDP cache data, needed for BC
    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None):
        # `ddp_cache_data` was originally added for compatibility with `torch.distributed` (DDP). See #36212
        # and #36373 for more information. In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the
        # iterable contains the key and value states for a layer gathered across replicas by torch.distributed
        # (shape=[global batch size, num_heads, seq_len, head_dim]).
        if ddp_cache_data is not None:
            layers = []
            for key_states, value_states, gate_states, state, sum_of_keys in ddp_cache_data:
                layers.append(DynamicLayer.from_tensors(key_states, value_states, gate_states, state, sum_of_keys))
            super().__init__(layers=layers)
        else:
            super().__init__(layer_class_to_replicate=DynamicLayer)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values, layer.gatings, layer.state, layer.sum_of_keys),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]) -> "Cache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states, gate_states, state, sum_of_keys = past_key_values[layer_idx]
                cache.update(key_states, value_states, gate_states, state, sum_of_keys, layer_idx)
        return cache


# Utilities for `DynamicCache` <> torch.export support

if is_torch_greater_or_equal("2.3"):

    def _get_cache_dict(cache: DynamicCache):
        if any(not isinstance(layer, DynamicLayer) for layer in cache.layers):
            raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

        if not is_torch_greater_or_equal_than_2_6:
            logger.warning_once(
                "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
            )

        return {
            "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
            "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
            "gating_cache": [layer.gatings for layer in cache.layers if layer.gatings is not None],
            "state_cache": [layer.state for layer in cache.layers if layer.state is not None],
            "sum_of_keys_cache": [layer.sum_of_keys for layer in cache.layers if layer.sum_of_keys is not None],
        }

    def _unflatten_dynamic_cache(
        values,
        context: torch.utils._pytree.Context,
    ):
        dictionary = torch.utils._pytree._dict_unflatten(values, context)
        cache = DynamicCache()
        # Reconstruct layers from keys and values lists
        key_list = dictionary.get("key_cache", [])
        value_list = dictionary.get("value_cache", [])
        gating_list = dictionary.get("gating_cache", [])
        state_list = dictionary.get("state_cache", [])
        sum_of_keys_list = dictionary.get("sum_of_keys_cache", [])
        for idx in range(max(len(key_list), len(value_list), len(gating_list), len(state_list), len(sum_of_keys_list))):
            key = key_list[idx] if idx < len(key_list) else None
            value = value_list[idx] if idx < len(value_list) else None
            gating = gating_list[idx] if idx < len(gating_list) else None
            state = state_list[idx] if idx < len(state_list) else None
            sum_of_keys = sum_of_keys_list[idx] if idx < len(sum_of_keys_list) else None
            cache.update(key, value, gating, state, sum_of_keys, idx)
        return cache

    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
        _unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
            _get_cache_dict(dynamic_cache)
        ),
    )
    # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
    torch.fx._pytree.register_pytree_flatten_spec(
        DynamicCache, lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec)
    )