from types import SimpleNamespace

import torch

from prime_rl.inference.patches import _patch_vllm_029_nemotron


class _BaseLayerWithLoRA:
    def __init__(self, base_layer):
        self.base_layer = base_layer


class _NemotronHForCausalLM:
    is_non_gated_moe = True
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    embedding_modules = {"embed_tokens": "input_embeddings"}
    lora_skip_prefixes = ["mtp."]


class _VisionModel:
    def __init__(self, output):
        self.output = output
        self.input_dtypes = []

    def __call__(self, pixel_values, **_kwargs):
        self.input_dtypes.append(pixel_values.dtype)
        return None, self.output.clone()


class _NemotronVL:
    pass


def _make_nano_module():
    class Processor:
        @staticmethod
        def get_video_repl(**_kwargs):
            return SimpleNamespace(full=[1, 2, 3])

    def merge_embeddings(*, inputs_embeds, multimodal_embeddings, is_multimodal):
        return inputs_embeds, multimodal_embeddings, is_multimodal

    return SimpleNamespace(
        NemotronH_Nano_VL_V2=_NemotronVL,
        NanoNemotronVLProcessor=Processor,
        cached_tokenizer_from_config=lambda _config: object(),
        _merge_multimodal_embeddings=merge_embeddings,
    )


def test_patch_enables_language_model_lora_and_is_idempotent():
    nano = _make_nano_module()

    _patch_vllm_029_nemotron(nano, _NemotronHForCausalLM, _BaseLayerWithLoRA)
    patched_extract = _NemotronVL.extract_feature
    _patch_vllm_029_nemotron(nano, _NemotronHForCausalLM, _BaseLayerWithLoRA)

    assert _NemotronVL.supports_lora is True
    assert _NemotronVL.is_non_gated_moe is True
    assert _NemotronVL.packed_modules_mapping == _NemotronHForCausalLM.packed_modules_mapping
    assert _NemotronVL.embedding_modules == _NemotronHForCausalLM.embedding_modules
    assert _NemotronVL.lora_skip_prefixes == ["mtp."]
    assert _NemotronVL.extract_feature is patched_extract


def test_patch_aligns_fixed_and_dynamic_vision_compute_dtype():
    nano = _make_nano_module()
    _patch_vllm_029_nemotron(nano, _NemotronHForCausalLM, _BaseLayerWithLoRA)

    model = object.__new__(_NemotronVL)
    model.llm_dtype = torch.float16
    model.patch_size = 1
    model.downsample_ratio = 1.0
    model.video_temporal_patch_size = 1
    model.pixel_shuffle = lambda value, scale_factor: value
    model.pixel_shuffle_dynamic_res = lambda value, imgs_sizes: value
    model.mlp1 = lambda value: value

    model.vision_model = _VisionModel(torch.ones(2, 4, 3, dtype=torch.float32))
    fixed = model.extract_feature(torch.ones(2, 3, 2, 2, dtype=torch.float32))
    assert model.vision_model.input_dtypes == [torch.float16]
    assert fixed.dtype == torch.float16

    model.vision_model = _VisionModel(torch.ones(1, 4, 3, dtype=torch.float32))
    dynamic = model.extract_feature_dynamic(
        torch.ones(1, 3, 2, 2, dtype=torch.float32),
        imgs_sizes=[(2, 2)],
    )
    assert model.vision_model.input_dtypes == [torch.float16]
    assert dynamic.dtype == torch.float16


def test_patch_uses_base_embeddings_for_video_indicator_tokens():
    nano = _make_nano_module()
    _patch_vllm_029_nemotron(nano, _NemotronHForCausalLM, _BaseLayerWithLoRA)

    embedded_token_ids = []

    def embed_tokens(token_ids):
        embedded_token_ids.append(token_ids.tolist())
        return torch.ones(len(token_ids), 2)

    model = object.__new__(_NemotronVL)
    model.model_config = object()
    model._img_start_token_ids = [10]
    model._img_end_token_ids = [11]
    model._img_context_token_ids = [2]
    model.get_language_model = lambda: SimpleNamespace(
        model=SimpleNamespace(embed_tokens=_BaseLayerWithLoRA(embed_tokens))
    )

    inputs, video, mask = model._create_final_video_embeddings(
        torch.ones(1, 2),
        num_tokens_per_frame=[1],
        frames_indices=[0],
        frame_duration_ms=100,
    )

    assert embedded_token_ids == [[1, 2, 3]]
    assert inputs.shape == (3, 2)
    assert video.shape == (1, 2)
    assert mask.tolist() == [False, True, False]
