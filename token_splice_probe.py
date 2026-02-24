#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from typing import Any

import httpx

try:
    from prime_rl.inference.config import MODEL_TOOL_CALL_PARSER
except Exception:
    MODEL_TOOL_CALL_PARSER = {}


TOOLS_V1: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

TOOLS_V2: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Compute a+b and return only the integer result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "b": {"type": "integer"},
                    "a": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


class HTTPFailure(RuntimeError):
    def __init__(self, method: str, path: str, status: int, body: str):
        super().__init__(f"{method} {path} -> HTTP {status}: {body}")
        self.method = method
        self.path = path
        self.status = status
        self.body = body


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_signature(tools: list[dict[str, Any]] | None) -> str:
    return hashlib.sha256(canonical_json(tools or []).encode("utf-8")).hexdigest()


def normalize_base_url(raw: str) -> tuple[str, str]:
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        return raw, raw[:-3]
    return f"{raw}/v1", raw


def auth_headers(api_key_env: str) -> dict[str, str]:
    api_key = os.getenv(api_key_env, "EMPTY")
    if not api_key or api_key == "EMPTY":
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def short_text(text: str, n: int = 700) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = await client.request(method, path, json=payload)
    body = short_text(response.text.strip())
    if response.status_code >= 400:
        raise HTTPFailure(method, path, response.status_code, body)
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {path} -> invalid JSON response: {body}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{method} {path} -> JSON response is not an object")
    return data


async def get_models(v1_client: httpx.AsyncClient) -> list[str]:
    data = await request_json(v1_client, "GET", "/models")
    models = data.get("data")
    if not isinstance(models, list):
        return []
    result: list[str] = []
    for item in models:
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str):
                result.append(model_id)
    return result


def make_chat_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tokens: list[int] | None = None,
    tool_choice: Any | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "return_token_ids": True,
    }
    if tools is not None:
        body["tools"] = tools
    if tokens is not None:
        body["tokens"] = tokens
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return body


async def chat_completion(
    v1_client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
) -> dict[str, Any]:
    body = make_chat_body(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )
    return await request_json(v1_client, "POST", "/chat/completions", body)


async def chat_completion_tokens(
    v1_client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, Any]],
    tokens: list[int],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body = make_chat_body(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        tools=tools,
        tokens=tokens,
    )
    return await request_json(v1_client, "POST", "/chat/completions/tokens", body)


async def tokenize_messages(
    root_client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool | None = None,
) -> list[int]:
    body: dict[str, Any] = {"model": model, "messages": messages}
    if tools is not None:
        body["tools"] = tools
    if add_generation_prompt is not None:
        body["add_generation_prompt"] = add_generation_prompt
    tokenized = await request_json(root_client, "POST", "/tokenize", body)
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise RuntimeError("/tokenize response missing integer tokens list")
    return tokens


def normalize_assistant_message(raw_message: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": raw_message.get("content"),
    }
    for key in ("tool_calls", "reasoning_content", "thinking_blocks"):
        if key in raw_message and raw_message[key] is not None:
            msg[key] = raw_message[key]
    return msg


def extract_model_prefix_ids(chat_resp: dict[str, Any]) -> tuple[list[int] | None, str | None]:
    choices = chat_resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "missing choices"
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None, "choice[0] is not an object"

    prompt_ids = chat_resp.get("prompt_token_ids")
    completion_ids = choice0.get("token_ids")

    if not isinstance(prompt_ids, list) or not all(isinstance(t, int) for t in prompt_ids):
        return None, "missing prompt_token_ids"
    if not isinstance(completion_ids, list) or not all(isinstance(t, int) for t in completion_ids):
        return None, "missing completion token_ids"

    return prompt_ids + completion_ids, None


def strict_replace_tokens(
    model_prefix_ids: list[int],
    template_prefix_ids: list[int],
    full_ids: list[int],
) -> tuple[list[int] | None, str | None]:
    if len(template_prefix_ids) > len(full_ids):
        return None, "template_prefix longer than full_ids"
    if full_ids[: len(template_prefix_ids)] != template_prefix_ids:
        return None, "template_prefix is not an exact prefix of full_ids"
    return model_prefix_ids + full_ids[len(template_prefix_ids) :], None


async def verify_super_path(
    v1_client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    try:
        resp = await chat_completion(v1_client, model, messages, max_tokens=16, tools=tools)
    except Exception as exc:
        return False, str(exc)
    finish_reason = ((resp.get("choices") or [{}])[0] or {}).get("finish_reason")
    return True, f"super call worked (finish_reason={finish_reason})"


async def scenario_linear_strict_splice(
    v1_client: httpx.AsyncClient,
    root_client: httpx.AsyncClient,
    model: str,
) -> dict[str, Any]:
    turn1 = [{"role": "user", "content": "Reply with exactly BLUE and nothing else."}]
    turn2_user = {"role": "user", "content": "Reply with exactly GREEN and nothing else."}

    try:
        first = await chat_completion(v1_client, model, turn1, max_tokens=32)
    except Exception as exc:
        return {"status": "fail", "reason": f"turn1 chat failed: {exc}"}

    choices = first.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {"status": "fail", "reason": "turn1 response missing valid choices[0]"}
    choice0 = choices[0]

    raw_message = choice0.get("message")
    if not isinstance(raw_message, dict):
        return {"status": "fail", "reason": "turn1 response missing assistant message"}
    assistant_msg = normalize_assistant_message(raw_message)
    turn2 = turn1 + [assistant_msg, turn2_user]

    async def fallback(reason: str) -> dict[str, Any]:
        ok, detail = await verify_super_path(v1_client, model, turn2)
        if ok:
            return {"status": "fallback", "reason": reason, "fallback": detail}
        return {"status": "fail", "reason": reason, "fallback": detail}

    model_prefix_ids, prefix_reason = extract_model_prefix_ids(first)
    if model_prefix_ids is None:
        return await fallback(f"missing model prefix ids: {prefix_reason}")

    if choice0.get("finish_reason") == "length":
        return await fallback("prior step is truncated (finish_reason=length)")

    try:
        full_ids = await tokenize_messages(root_client, model, turn2)
        template_prefix_ids = await tokenize_messages(
            root_client,
            model,
            turn1 + [assistant_msg],
            add_generation_prompt=False,
        )
    except Exception as exc:
        return await fallback(f"tokenization failed: {exc}")

    spliced_ids, replace_reason = strict_replace_tokens(model_prefix_ids, template_prefix_ids, full_ids)
    if spliced_ids is None:
        return await fallback(f"strict replace rejected: {replace_reason}")

    try:
        token_resp = await chat_completion_tokens(v1_client, model, turn2, spliced_ids, max_tokens=16)
    except Exception as exc:
        return await fallback(f"/chat/completions/tokens failed: {exc}")

    try:
        normal_resp = await chat_completion(v1_client, model, turn2, max_tokens=16)
    except Exception as exc:
        return {"status": "fail", "reason": f"normal turn2 chat failed after token route succeeded: {exc}"}

    token_choice = (token_resp.get("choices") or [{}])[0] or {}
    normal_choice = (normal_resp.get("choices") or [{}])[0] or {}

    token_content = (token_choice.get("message") or {}).get("content")
    normal_content = (normal_choice.get("message") or {}).get("content")

    echoed_prompt_ids = token_resp.get("prompt_token_ids")
    echoed_match = isinstance(echoed_prompt_ids, list) and echoed_prompt_ids == spliced_ids

    status = "pass"
    warnings: list[str] = []
    if token_content != normal_content:
        status = "warn"
        warnings.append("token route content differs from normal route at temperature=0")
    if isinstance(echoed_prompt_ids, list) and not echoed_match:
        status = "warn"
        warnings.append("server prompt_token_ids differ from submitted spliced tokens")

    return {
        "status": status,
        "reason": "strict splice executed",
        "warnings": warnings,
        "spliced_len": len(spliced_ids),
        "full_ids_len": len(full_ids),
        "template_prefix_len": len(template_prefix_ids),
        "model_prefix_len": len(model_prefix_ids),
        "token_content": token_content,
        "normal_content": normal_content,
    }


async def scenario_prefix_ends_with_tool_call(
    root_client: httpx.AsyncClient,
    model: str,
) -> dict[str, Any]:
    prefix_bad = [
        {"role": "user", "content": "Call the add tool with a=2 and b=3."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                }
            ],
        },
    ]
    full_valid = prefix_bad + [
        {"role": "tool", "tool_call_id": "call_1", "content": "5"},
        {"role": "user", "content": "Now reply with exactly DONE."},
    ]

    full_ok = False
    prefix_ok = False
    full_err = None
    prefix_err = None
    full_ids: list[int] = []
    prefix_ids: list[int] = []

    try:
        full_ids = await tokenize_messages(root_client, model, full_valid, tools=TOOLS_V1)
        full_ok = True
    except Exception as exc:
        full_err = str(exc)

    try:
        prefix_ids = await tokenize_messages(
            root_client,
            model,
            prefix_bad,
            tools=TOOLS_V1,
            add_generation_prompt=False,
        )
        prefix_ok = True
    except Exception as exc:
        prefix_err = str(exc)

    if not full_ok:
        return {
            "status": "skip",
            "reason": "full tool sequence itself is not tokenizable on this model/template",
            "full_error": full_err,
        }

    if not prefix_ok:
        return {
            "status": "warn",
            "reason": "prefix ending with assistant tool_calls is not tokenizable; strict path must fallback",
            "full_tokenize_ok": True,
            "prefix_tokenize_ok": False,
            "prefix_error": prefix_err,
        }

    prefix_is_prefix = len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids
    if not prefix_is_prefix:
        return {
            "status": "warn",
            "reason": "prefix tokenized but does not align as exact prefix of full sequence",
            "full_len": len(full_ids),
            "prefix_len": len(prefix_ids),
        }

    return {
        "status": "pass",
        "reason": "prefix ending with assistant tool_calls tokenizes and aligns",
        "full_len": len(full_ids),
        "prefix_len": len(prefix_ids),
    }


async def scenario_think_prefix_compaction(
    root_client: httpx.AsyncClient,
    model: str,
) -> dict[str, Any]:
    prefix_messages = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "<think>R1</think>A1"},
    ]
    full_messages = prefix_messages + [{"role": "user", "content": "U2"}]

    try:
        prefix_ids = await tokenize_messages(
            root_client,
            model,
            prefix_messages,
            add_generation_prompt=False,
        )
        full_ids = await tokenize_messages(root_client, model, full_messages)
    except Exception as exc:
        return {"status": "skip", "reason": f"tokenize failed for think-prefix scenario: {exc}"}

    prefix_is_prefix = len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids
    if prefix_is_prefix:
        return {
            "status": "pass",
            "reason": "think-tag prefix still aligns for this model/template",
            "prefix_len": len(prefix_ids),
            "full_len": len(full_ids),
        }
    return {
        "status": "warn",
        "reason": "think-tag prefix is not an exact prefix of full tokenization (compaction/rewrite risk)",
        "prefix_len": len(prefix_ids),
        "full_len": len(full_ids),
    }


async def scenario_tool_schema_drift(
    root_client: httpx.AsyncClient,
    model: str,
) -> dict[str, Any]:
    msgs = [{"role": "user", "content": "Just say hi."}]
    sig_v1 = tool_signature(TOOLS_V1)
    sig_v2 = tool_signature(TOOLS_V2)

    ids_v1: list[int] | None = None
    ids_v2: list[int] | None = None
    err_v1 = None
    err_v2 = None

    try:
        ids_v1 = await tokenize_messages(root_client, model, msgs, tools=TOOLS_V1)
    except Exception as exc:
        err_v1 = str(exc)

    try:
        ids_v2 = await tokenize_messages(root_client, model, msgs, tools=TOOLS_V2)
    except Exception as exc:
        err_v2 = str(exc)

    if ids_v1 is None and ids_v2 is None:
        return {
            "status": "skip",
            "reason": "tool-schema tokenization probe not supported on this model/template",
            "err_v1": err_v1,
            "err_v2": err_v2,
        }

    if ids_v1 is None or ids_v2 is None:
        return {
            "status": "warn",
            "reason": "one tool schema tokenizes and the other fails; strict guard should fallback on signature mismatch",
            "sig_v1": sig_v1,
            "sig_v2": sig_v2,
            "err_v1": err_v1,
            "err_v2": err_v2,
        }

    tokens_changed = ids_v1 != ids_v2
    if tokens_changed:
        return {
            "status": "warn",
            "reason": "same messages tokenize differently under tool schema drift",
            "sig_v1": sig_v1,
            "sig_v2": sig_v2,
            "len_v1": len(ids_v1),
            "len_v2": len(ids_v2),
        }

    return {
        "status": "pass",
        "reason": "tool signatures differ but tokenization stayed identical for this prompt",
        "sig_v1": sig_v1,
        "sig_v2": sig_v2,
    }


async def scenario_truncated_turn_guard(
    v1_client: httpx.AsyncClient,
    model: str,
) -> dict[str, Any]:
    msgs = [
        {
            "role": "user",
            "content": "Write at least 120 words about oceans, coral reefs, weather systems, and shipping lanes.",
        }
    ]
    try:
        resp = await chat_completion(v1_client, model, msgs, max_tokens=1)
    except Exception as exc:
        return {"status": "fail", "reason": f"truncation probe failed: {exc}"}

    choice0 = (resp.get("choices") or [{}])[0] or {}
    finish = choice0.get("finish_reason")
    if finish == "length":
        return {
            "status": "pass",
            "reason": "prior-turn truncation is observable; strict guard should fallback",
        }
    return {
        "status": "warn",
        "reason": f"did not observe finish_reason=length (got {finish}); truncation guard not exercised on this run",
    }


def collect_watchouts(scenarios: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []

    linear = scenarios.get("linear_strict_splice", {})
    if linear.get("status") in ("fallback", "warn", "fail"):
        out.append(f"Linear strict splice did not cleanly pass ({linear.get('status')}): {linear.get('reason')}")

    tc = scenarios.get("prefix_ends_with_tool_call", {})
    if tc.get("status") == "warn":
        out.append("Prefix ending with assistant tool_calls can fail /tokenize; fallback to super is mandatory.")

    think = scenarios.get("think_prefix_compaction", {})
    if think.get("status") == "warn":
        out.append("Think-tag/history compaction can break exact prefix alignment; fallback path required.")

    drift = scenarios.get("tool_schema_drift", {})
    if drift.get("status") == "warn":
        out.append("Tool schema drift changes tokenization behavior; enforce tool-signature match before splicing.")

    trunc = scenarios.get("truncated_turn_guard", {})
    if trunc.get("status") == "warn":
        out.append("Could not confirm truncation guard on this run; keep explicit finish_reason=length fallback.")

    return out


async def probe_model(
    v1_client: httpx.AsyncClient,
    root_client: httpx.AsyncClient,
    endpoint_label: str,
    model: str,
) -> dict[str, Any]:
    parser = MODEL_TOOL_CALL_PARSER.get(model)
    scenarios: dict[str, dict[str, Any]] = {}

    scenarios["linear_strict_splice"] = await scenario_linear_strict_splice(v1_client, root_client, model)
    scenarios["prefix_ends_with_tool_call"] = await scenario_prefix_ends_with_tool_call(root_client, model)
    scenarios["think_prefix_compaction"] = await scenario_think_prefix_compaction(root_client, model)
    scenarios["tool_schema_drift"] = await scenario_tool_schema_drift(root_client, model)
    scenarios["truncated_turn_guard"] = await scenario_truncated_turn_guard(v1_client, model)

    return {
        "base_url": endpoint_label,
        "model": model,
        "tool_call_parser": parser,
        "scenarios": scenarios,
        "watchouts": collect_watchouts(scenarios),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe strict token-splice behavior against Prime-RL inference endpoints."
    )
    parser.add_argument(
        "--base-urls",
        nargs="+",
        required=True,
        help="One or more inference base URLs (with or without /v1).",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit model IDs to probe.")
    parser.add_argument("--max-models-per-endpoint", type=int, default=0, help="Limit per endpoint (0 = no limit).")
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Include models not in MODEL_TOOL_CALL_PARSER map.",
    )
    parser.add_argument("--api-key-env", default="VLLM_API_KEY", help="Env var for API key (default: VLLM_API_KEY).")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    parser.add_argument(
        "--output",
        default=f"token_splice_probe_{int(time.time())}.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("none", "fail", "warn"),
        default="fail",
        help="Exit non-zero on: none, fail-only, or fail+warn+fallback.",
    )
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    runs: list[dict[str, Any]] = []
    all_statuses: list[str] = []

    headers = auth_headers(args.api_key_env)
    timeout = httpx.Timeout(args.timeout)

    for raw in args.base_urls:
        v1_url, root_url = normalize_base_url(raw)
        print(f"\n[endpoint] {raw} -> v1={v1_url} root={root_url}")

        async with (
            httpx.AsyncClient(base_url=v1_url, headers=headers, timeout=timeout) as v1_client,
            httpx.AsyncClient(base_url=root_url, headers=headers, timeout=timeout) as root_client,
        ):
            try:
                discovered = await get_models(v1_client)
            except Exception as exc:
                print(f"  failed to list /v1/models: {exc}")
                continue

            if args.models:
                selected = [m for m in args.models if m in discovered]
                missing = [m for m in args.models if m not in discovered]
                if missing:
                    print(f"  requested models not present on endpoint: {missing}")
            else:
                selected = list(discovered)

            if MODEL_TOOL_CALL_PARSER and not args.include_unsupported:
                selected = [m for m in selected if m in MODEL_TOOL_CALL_PARSER]

            if args.max_models_per_endpoint > 0:
                selected = selected[: args.max_models_per_endpoint]

            print(f"  discovered={len(discovered)} selected={len(selected)}")

            for model in selected:
                print(f"  probing model: {model}")
                result = await probe_model(v1_client, root_client, raw, model)
                runs.append(result)
                for scenario_result in result["scenarios"].values():
                    all_statuses.append(scenario_result.get("status", "fail"))

    status_counts: dict[str, int] = {}
    for status in all_statuses:
        status_counts[status] = status_counts.get(status, 0) + 1

    report = {
        "generated_at_unix": int(time.time()),
        "base_urls": args.base_urls,
        "supported_model_map_size": len(MODEL_TOOL_CALL_PARSER),
        "include_unsupported": args.include_unsupported,
        "runs": runs,
        "summary": {
            "models_probed": len(runs),
            "scenario_status_counts": status_counts,
            "models_with_watchouts": [
                {"model": r["model"], "base_url": r["base_url"], "watchouts": r["watchouts"]}
                for r in runs
                if r.get("watchouts")
            ],
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Token Splice Probe Summary ===")
    print(f"report: {args.output}")
    print(f"models_probed: {len(runs)}")
    print(f"scenario_status_counts: {status_counts}")

    for item in report["summary"]["models_with_watchouts"]:
        print(f"\n[watchouts] {item['model']} @ {item['base_url']}")
        for message in item["watchouts"]:
            print(f"  - {message}")

    fail_count = status_counts.get("fail", 0)
    warn_like_count = fail_count + status_counts.get("warn", 0) + status_counts.get("fallback", 0)

    if args.fail_on == "none":
        return 0
    if args.fail_on == "fail":
        return 1 if fail_count > 0 else 0
    return 1 if warn_like_count > 0 else 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run())
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)
