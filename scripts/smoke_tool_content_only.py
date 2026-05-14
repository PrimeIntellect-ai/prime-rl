"""Smoke test for `RoleLossMaskConfig.tool_content_only`.

Builds a synthetic chat trajectory (system + user + assistant w/ tool_call +
tool_response + assistant), tokenizes via the real Qwen3-4B-Instruct tokenizer,
and checks the per-token mask in two configurations:

  (1) `tool_content_only=False` — should mark the *entire* tool span (header,
      envelope tags, content, footer) — preserves prior behavior.
  (2) `tool_content_only=True`  — should mark only the *content* tokens; the
      `<|im_start|>user\\n<tool_response>\\n…\\n</tool_response><|im_end|>\\n`
      envelope is excluded.

Run from prime-rl root:
    uv run python scripts/smoke_tool_content_only.py
"""

from __future__ import annotations

from transformers import AutoTokenizer

from prime_rl.configs.trainer import RoleLossMaskConfig
from prime_rl.orchestrator.trajectories import _build_role_loss_mask_from_token_stream


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    messages = [
        {"role": "system", "content": "You are a Forth assistant."},
        {"role": "user", "content": "Define `dbl`."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_code", "arguments": '{"code": ": dbl dup + ;"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "stack: [4]"},
        {"role": "assistant", "content": "Looks good. Submitting."},
    ]

    # apply_chat_template(tokenize=True) returns an Encoding on this transformers
    # version — render to text first and encode separately to get a flat id list.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    n = len(full_ids)

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tool_response = tokenizer.convert_tokens_to_ids("<tool_response>")
    tool_response_close = tokenizer.convert_tokens_to_ids("</tool_response>")
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    # Locate the (only) tool span in the rendered trajectory.
    try:
        open_idx = full_ids.index(tool_response)
        close_idx = full_ids.index(tool_response_close, open_idx + 1)
    except ValueError as e:
        raise AssertionError(f"Tool span not found in trajectory: {e}")

    # Header: from the <|im_start|> immediately preceding <tool_response> through
    # the token just before <tool_response>. Footer: from <|im_end|> immediately
    # after </tool_response> through its trailing \n.
    span_start = open_idx
    while span_start >= 0 and full_ids[span_start] != im_start:
        span_start -= 1
    assert span_start >= 0, "no <|im_start|> before <tool_response>"
    span_end = close_idx
    while span_end < n and full_ids[span_end] != im_end:
        span_end += 1
    assert span_end < n, "no <|im_end|> after </tool_response>"

    header_idx = list(range(span_start, open_idx))                # <|im_start|>, "user", \n
    open_env_idx = [open_idx]                                     # <tool_response>
    if open_idx + 1 < n and full_ids[open_idx + 1] == newline_id:
        open_env_idx.append(open_idx + 1)
    content_idx = list(range(open_env_idx[-1] + 1, close_idx))
    close_env_idx = [close_idx]
    if close_idx - 1 >= 0 and full_ids[close_idx - 1] == newline_id:
        close_env_idx.insert(0, close_idx - 1)
        if content_idx and content_idx[-1] == close_idx - 1:
            content_idx = content_idx[:-1]
    footer_idx = [span_end]
    if span_end + 1 < n and full_ids[span_end + 1] == newline_id:
        footer_idx.append(span_end + 1)
    envelope_idx = header_idx + open_env_idx + close_env_idx + footer_idx

    print(f"Trajectory: {n} tokens, tool span at [{span_start}..{span_end+1}]")
    print(f"  envelope: {len(envelope_idx)} tokens; content: {len(content_idx)} tokens")
    print(f"  content text: {tokenizer.decode([full_ids[p] for p in content_idx])!r}")
    print()

    cfg_full = RoleLossMaskConfig(assistant=False, tool=True, tool_content_only=False)
    mask_full = _build_role_loss_mask_from_token_stream(full_ids, tokenizer, cfg_full)
    for p in envelope_idx + content_idx:
        assert mask_full[p] is True, (
            f"[FAIL] content_only=False: pos {p} ({tokenizer.decode([full_ids[p]])!r}) "
            f"expected True, got {mask_full[p]}"
        )
    print(f"[OK]  content_only=False: all {len(envelope_idx) + len(content_idx)} "
          f"tool-span tokens are True (envelope + content).")

    cfg_content = RoleLossMaskConfig(assistant=False, tool=True, tool_content_only=True)
    mask_content = _build_role_loss_mask_from_token_stream(full_ids, tokenizer, cfg_content)

    for p in envelope_idx:
        assert mask_content[p] is False, (
            f"[FAIL] content_only=True: envelope pos {p} "
            f"({tokenizer.decode([full_ids[p]])!r}) expected False, got True"
        )
    print(f"[OK]  content_only=True: all {len(envelope_idx)} envelope tokens are False.")
    for p in content_idx:
        assert mask_content[p] is True, (
            f"[FAIL] content_only=True: content pos {p} "
            f"({tokenizer.decode([full_ids[p]])!r}) expected True, got False"
        )
    print(f"[OK]  content_only=True: all {len(content_idx)} content tokens are True.")

    outside = [p for p in range(n) if p < span_start or p > footer_idx[-1]]
    out_true = [p for p in outside if mask_content[p]]
    assert not out_true, (
        f"[FAIL] content_only=True: {len(out_true)} non-tool tokens leaked True: "
        f"{[tokenizer.decode([full_ids[p]]) for p in out_true[:5]]}"
    )
    print(f"[OK]  content_only=True: no non-tool tokens leaked into the mask "
          f"({len(outside)} positions checked).")

    print()
    print("  header tokens (dropped under content_only):  ",
          [tokenizer.decode([full_ids[p]]) for p in header_idx])
    print("  opening envelope (dropped):                  ",
          [tokenizer.decode([full_ids[p]]) for p in open_env_idx])
    print("  closing envelope (dropped):                  ",
          [tokenizer.decode([full_ids[p]]) for p in close_env_idx])
    print("  footer tokens (dropped):                     ",
          [tokenizer.decode([full_ids[p]]) for p in footer_idx])
    print()
    print("[ALL OK]  tool_content_only mask behaves as specified.")


if __name__ == "__main__":
    main()
