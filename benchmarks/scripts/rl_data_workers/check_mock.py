import os, time, torch
from transformers import AutoProcessor
from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.trainer.rl.mock_mm import MockMMMicroBatches
torch.set_num_threads(1)
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
adapter = get_multimodal_adapter("qwen3_5")
for px in (512, 1024, 1440):
    os.environ["PRL_MOCK_MM_IMAGE_PX"] = str(px)
    mock = MockMMMicroBatches(16384)
    mb = mock.micro_batch(torch.Generator().manual_seed(0))
    refs = mb["mm_refs"]; tt = mb["mm_token_type_ids"][0]; ids = mb["input_ids"][0]
    assert all(bool((tt[r.offset:r.offset + r.length] == 1).all()) and int(ids[r.offset - 1]) == 248053 for r in refs.images)
    assert int(tt.sum()) == sum(r.length for r in refs.images)
    t0 = time.perf_counter(); out = materialize_mm_refs(refs, proc, adapter); dt = time.perf_counter() - t0
    print(f"{px}px: tokens/micro batch {ids.numel()}, samples {len(mb['seq_lens'])}, images {len(refs.images)}, image tokens {int(tt.sum())}, materialize {dt*1e3:.0f} ms on 1 thread")
