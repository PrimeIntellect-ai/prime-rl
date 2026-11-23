"""Find trivially detectable agent errors in rlm rollouts (candidates for early stop / negative reward).

Usage: python trivial_errors.py <env_prefix> <run_dir> [<run_dir> ...]
"""
import collections, glob, io, json, re, statistics as st, sys
import zstandard

NATIVE = {"ipython", "wait"}
NET = re.compile(r"\b(SSLError|SSLEOFError|URLError|ConnectError|ConnectionError|NameResolutionError|MaxRetryError)\b")

def checks_for_step(call_name, code, out):
    """Return the check names a single (tool call, tool result) step trips."""
    hit = []
    if call_name not in NATIVE:
        if re.fullmatch(r"ip\w*on|ipy\w*", call_name or ""):
            hit.append("A1 native call: typo of ipython")
        elif (call_name or "").startswith("rlm"):
            hit.append("A2 native call: rlm.* API as a tool")
        else:
            hit.append("A3 native call: other unknown tool")
    if re.search(r"'function' object has no attribute '(query|search|name|call|run)'", out):
        hit.append("B guessed method on an MCP tool (tool.query(...))")
    if re.search(r"unknown MCP tool|MCP tool '[^']+' is unavailable|has no attribute 'browsecomp_plus_search'|name 'browsecomp_plus_search' is not defined", out):
        hit.append("C wrong MCP tool name/id")
    if "invalid broker request" in out:
        hit.append("D invalid rlm API call (e.g. empty tools.search query)")
    if NET.search(out) and ("http" in code or "requests" in code or "urllib" in code or "curl" in code):
        hit.append("E internet access attempt (network is disabled)")
    if "'coroutine' object is not callable" in out or "was never awaited" in out:
        hit.append("G missing await")
    if re.search(r"'AgentHandle' object has no attribute|'AgentResult' object has no attribute", out):
        hit.append("K guessed subagent API attribute")
    if re.search(r"^SyntaxError|^IndentationError", out, re.M):
        hit.append("S SyntaxError in cell")
    return hit

def rollouts(prefix, runs):
    for run in runs:
        for p in sorted(glob.glob(f"{run}/monitors/file/traces/stream/*")):
            fh = open(p, "rb")
            s = zstandard.ZstdDecompressor().stream_reader(fh) if p.endswith(".zst") else fh
            for line in io.TextIOWrapper(s):
                d = json.loads(line)
                if d["env"]["name"].startswith(prefix) and d.get("traces"):
                    yield d

prefix, runs = sys.argv[1], sys.argv[2:]
rows = []
for d in rollouts(prefix, runs):
    t = d["traces"][0]
    nodes = t["nodes"]
    first, count = {}, collections.Counter()
    step, prev_args, repeat = 0, None, 0
    for i, nd in enumerate(nodes):
        m = nd["message"]
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        step += 1
        tc = m["tool_calls"][0]
        name, args = tc.get("name"), tc.get("arguments") or ""
        try:
            code = json.loads(args).get("code", "") if args else ""
        except Exception:
            code = ""
        out = next((str(n["message"].get("content")) for n in nodes[i + 1:] if n["message"]["role"] == "tool"), "")
        hits = checks_for_step(name, code, out)
        repeat = repeat + 1 if args == prev_args else 0
        prev_args = args
        if repeat == 2:
            hits.append("J same tool call 3x in a row")
        for h in hits:
            count[h] += 1
            first.setdefault(h, step)
    last = t.get("last_reply") or ""
    if not last.strip():
        count["H empty final reply"] += 1
        first.setdefault("H empty final reply", step)
    elif "exact answer" not in last.lower():
        count["I final reply without 'Exact Answer:'"] += 1
        first.setdefault("I final reply without 'Exact Answer:'", step)
    rows.append(dict(env=d["env"]["name"], reward=t.get("reward") or 0.0, steps=max(step, 1), count=count, first=first,
                     timeout=bool(t.get("is_timeout"))))

n = len(rows)
names = sorted({k for r in rows for k in r["count"]})
base = st.mean(r["reward"] for r in rows)
print(f"{n} rollouts ({', '.join(f'{k}={v}' for k, v in sorted(collections.Counter(r['env'] for r in rows).items()))}); mean reward {base:.2f}\n")
print("check | rollouts hit | reward if hit | reward if not | mean hits/rollout when hit | first hit at % of rollout (median) | hit rollouts with reward 1")
for k in names:
    hit = [r for r in rows if r["count"][k]]
    miss = [r for r in rows if not r["count"][k]]
    frac = st.median(r["first"][k] / r["steps"] for r in hit)
    print(" | ".join([k, f"{len(hit)} ({100*len(hit)/n:.0f}%)", f"{st.mean(r['reward'] for r in hit):.2f}",
                      f"{st.mean(r['reward'] for r in miss):.2f}" if miss else "-", f"{st.mean(r['count'][k] for r in hit):.1f}",
                      f"{100*frac:.0f}%", f"{sum(r['reward'] >= 1 for r in hit)}"]))
