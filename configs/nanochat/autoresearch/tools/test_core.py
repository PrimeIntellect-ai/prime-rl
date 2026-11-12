"""CPU check: packed CORE (prime-rl port) == nanochat core_eval.evaluate_example on a random tiny nanochat model."""
import sys, importlib.util, random, json, os, torch
import torch.nn.functional as F
import os
sys.path.insert(0, os.environ["NANOCHAT_DIR"])  # karpathy/nanochat @ ff46300
import cpu_import as c
from transformers import PreTrainedTokenizerFast

spec = importlib.util.spec_from_file_location("ne", f"{c.PRIME}/src/prime_rl/trainer/pretrain/nanochat_eval.py")
ne = importlib.util.module_from_spec(spec); spec.loader.exec_module(ne)
from nanochat import core_eval as ref

tok = PreTrainedTokenizerFast.from_pretrained(os.environ["NANOCHAT_TOKENIZER"])
BOS = tok.bos_token_id
torch.manual_seed(0)
model = c.build(2, device="cpu", seq_len=2048)
model.initialize_weights()
with torch.no_grad():
    for b in model.model.layers:
        b.attn.c_proj.weight.normal_(0, 0.05); b.mlp.c_proj.weight.normal_(0, 0.05)
    model.lm_head.weight.normal_(0, 0.05)
model.eval()

def hidden(input_ids, position_ids, seq_lens):
    return model.model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens).last_hidden_state

def logits_of(h):
    l = h.float() @ model.lm_head.weight.float().t()
    return 15 * torch.tanh(l / 15)

def forward_logits(input_ids, position_ids, seq_lens):
    return logits_of(hidden(input_ids, position_ids, seq_lens))[0]

class RefTok:
    def get_bos_token_id(self): return BOS
    def __call__(self, prompts, prepend=None):
        return [[prepend] + tok.encode(p, add_special_tokens=False) for p in prompts]

class RefModel:
    def __call__(self, input_ids):
        B, T = input_ids.shape
        out = []
        for i in range(B):
            out.append(logits_of(hidden(input_ids[i:i+1], torch.arange(T)[None], torch.tensor([T]))))
        return torch.cat(out, 0)

bundle = os.environ["EVAL_BUNDLE"]
tasks, baselines = ne.load_core_tasks(bundle, 14)
pick = [t for t in tasks if t["label"] in ("hellaswag_zeroshot", "jeopardy", "winograd", "arc_easy", "squad", "copa", "lambada_openai")]
mismatch = 0; total = 0
for task in pick:
    meta = dict(task_type=task["task_type"], num_fewshot=task["num_fewshot"], continuation_delimiter=task["delimiter"])
    with torch.no_grad():
        refres = [ref.evaluate_example(i, RefModel(), RefTok(), task["data"], "cpu", meta) for i in range(len(task["data"]))]
        out = ne.run_core(forward_logits, tok, bundle, 6, torch.device("cpu"), pack_tokens=4096, log=lambda s: None) if False else None
    # ours, single task
    reqs = ne.build_requests([task], 0, 1, lambda s: tok.encode(s, add_special_tokens=False), BOS)
    packs = ne.pack_requests(reqs, 4096)
    ours = [None] * len(task["data"])
    with torch.no_grad():
        for members in packs:
            seqs = [t for r in members for t in reqs[r]["tokens"]]
            tot = sum(map(len, seqs)); row = max(4096, -(-tot // 4096) * 4096)
            if tot < row: seqs = seqs + [[BOS] * (row - tot)]
            ids = torch.tensor([t for s in seqs for t in s])[None]
            pos = torch.tensor([i for s in seqs for i in range(len(s))])[None]
            lg = forward_logits(ids, pos, torch.tensor([len(s) for s in seqs]))
            losses = F.cross_entropy(lg, torch.roll(ids[0], -1), reduction="none"); preds = lg.argmax(-1)
            off = 0
            for r in members:
                q = reqs[r]; L, P = [], []
                for t in q["tokens"]:
                    L.append(losses[off:off+len(t)]); P.append(preds[off:off+len(t)]); off += len(t)
                ours[q["idx"]] = ne.score_example(q["task_type"], q["gold"], L, P, q["tokens"], q["starts"], q["ends"])
    mm = sum(a != b for a, b in zip(refres, ours)); mismatch += mm; total += len(ours)
    print(task["label"], task["task_type"], "ref", refres, "ours", ours, "mismatch", mm)
print("TOTAL mismatch", mismatch, "of", total)
# full run_core smoke on 3 per task
res = ne.run_core(forward_logits, tok, bundle, 11, torch.device("cpu"), pack_tokens=4096, log=print)
print(json.dumps(res["centered_results"])[:300], res["core_metric"])
