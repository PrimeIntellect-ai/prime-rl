"""Scan rollout traces for failures that trivial checks can catch.

Usage: python scan_failures.py <run_dir> [env_prefix]
"""
import collections, glob, io, json, re, sys
import zstandard

run, prefix = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
NATIVE = {"ipython", "wait"}
EXC = re.compile(r"^(\w+(?:Error|Exception|Interrupt))\b", re.M)
checks = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(dict)
n_by_env = collections.Counter()

def flag(env, name, ex=None):
    checks[env][name] += 1
    if ex is not None and name not in examples[env]:
        examples[env][name] = str(ex)[:400]

for p in sorted(glob.glob(f"{run}/monitors/file/traces/stream/*")):
    fh = open(p, "rb")
    stream = zstandard.ZstdDecompressor().stream_reader(fh) if p.endswith(".zst") else fh
    for line in io.TextIOWrapper(stream):
        d = json.loads(line)
        env = d["env"]["name"]
        if not env.startswith(prefix):
            continue
        n_by_env[env] += 1
        if not d.get("traces"):
            flag(env, "no trace (harness error)"); continue
        t = d["traces"][0]
        hit = set()
        def once(name, ex=None):
            if name not in hit:
                hit.add(name); flag(env, name, ex)
        calls_seen = collections.Counter()
        for n in t["nodes"]:
            m = n["message"]
            c = m.get("content")
            c = c if isinstance(c, str) else json.dumps(c) if c else ""
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    args = tc.get("arguments") or tc.get("function", {}).get("arguments") or ""
                    if name not in NATIVE:
                        once(f"native call to unknown tool", name)
                    calls_seen[args] += 1
                    if calls_seen[args] == 3:
                        once("same tool call repeated 3+ times", args[:200])
                if m.get("tool_calls") and len(m["tool_calls"]) > 1:
                    once("several tool calls in one step")
                if not m.get("tool_calls") and not c.strip():
                    once("empty assistant message")
            elif m["role"] == "tool":
                if "web search is unavailable" in c:
                    once("search(): web search is unavailable")
                for e in set(EXC.findall(c)):
                    once(f"ipython exception: {e}", c[-300:])
                if re.search(r"\b(403|429)\b.*(Forbidden|Too Many|rate)", c, re.I) or "robot policy" in c:
                    once("HTTP 403/429/robot block in tool output", c[:200])
                if "unknown tool" in c.lower() or "not a valid tool" in c.lower() or "no such tool" in c.lower():
                    once("tool result says unknown tool", c[:200])
                if "timed out" in c.lower() or "TimeoutError" in c:
                    once("tool timeout", c[-200:])
                if "omitted" in c and "bytes" in c:
                    once("tool output truncated")
        last = t.get("last_reply") or ""
        if t.get("is_timeout"):
            once("rollout timeout")
        if not last.strip():
            once("empty final reply")
        if "<think>" in last or "</think>" in last:
            once("think tag in final reply")
        if t.get("stop_condition") not in ("agent_completed",):
            once(f"stop_condition={t.get('stop_condition')}")
        if env.startswith(("wideseek", "openseeker")) and last.strip():
            pipes = [l for l in last.splitlines() if l.strip().startswith("|")]
            if not pipes:
                once("final reply has no table")
            else:
                outside = [l for l in last.splitlines() if l.strip() and not l.strip().startswith(("|", "```"))]
                if outside:
                    once("final reply has text outside the table", outside[0][:150])
        if t.get("reward") == 0:
            once("reward 0")

for env in sorted(n_by_env):
    print(f"\n== {env} ({n_by_env[env]} rollouts)")
    for name, k in checks[env].most_common():
        ex = examples[env].get(name)
        print(f"  {k:4d} {100*k/n_by_env[env]:5.1f}%  {name}" + (f"   e.g. {ex[:160]!r}" if ex else ""))
