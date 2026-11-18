"""Replay a trace's first model call through the router with/without chat_template_kwargs effort=max; print prompt tokens."""
import json
import sys

import httpx

trace = json.load(open(sys.argv[1]))["trace"]
url = sys.argv[2]
messages = [trace["nodes"][0]["message"], trace["nodes"][1]["message"]]
tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}} for t in trace["nodes"][0]["tools"]]
print("trace first call prompt_tokens:", trace["calls"][0]["usage"]["prompt_tokens"])
for label, extra in [("no effort", {}), ("ctk max", {"chat_template_kwargs": {"reasoning_effort": "max"}}), ("ctk high", {"chat_template_kwargs": {"reasoning_effort": "high"}})]:
    body = {"model": "PrimeIntellect/DeepSeek-V4-Flash-0731-bf16", "messages": messages, "tools": tools, "max_tokens": 1, **extra}
    r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=120)
    print(label, r.status_code, r.json().get("usage", r.text[:200]))
