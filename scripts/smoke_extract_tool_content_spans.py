"""Smoke test for `_extract_tool_content_spans` — the tool-content-only
span extractor used by `capture_tool_logprobs` to locate per-tool token
ranges in a rollout's prompt for scoring.

Builds a synthetic chat trajectory with two tool calls of different
names (run_code, lookup_doc), renders+tokenizes via real Qwen3-4B
tokenizer, then asserts:

  * each tool message produces exactly one span,
  * spans are content-only (don't include the chat-template envelope),
  * span text decodes back to the intended tool content,
  * tool names are correctly mapped to spans.

Run from prime-rl root:
    uv run python scripts/smoke_extract_tool_content_spans.py
"""

from __future__ import annotations

from transformers import AutoTokenizer

from prime_rl.orchestrator.trajectories import (
    _extract_tool_content_spans,
    _tool_call_names_in_order,
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    messages = [
        {"role": "system", "content": "You are a Forth assistant."},
        {"role": "user", "content": "Define `dbl`."},
        {
            "role": "assistant",
            "content": "Let me try it.",
            "tool_calls": [
                {
                    "id": "call_run_1",
                    "type": "function",
                    "function": {"name": "run_code", "arguments": '{"code": ": dbl dup + ;"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_run_1", "content": "stack: [4]"},
        {
            "role": "assistant",
            "content": "Let me check what dup does.",
            "tool_calls": [
                {
                    "id": "call_lkp_1",
                    "type": "function",
                    "function": {"name": "lookup_doc", "arguments": '{"word": "dup"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_lkp_1", "content": "duplicates the top of stack"},
        {"role": "assistant", "content": "Looks right, done."},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

    tool_call_names = _tool_call_names_in_order(prompt=messages[:1], completion=messages[1:])
    print(f"resolved tool_call_names_in_order = {tool_call_names}")
    assert tool_call_names == ["run_code", "lookup_doc"], (
        f"expected ['run_code', 'lookup_doc'], got {tool_call_names}"
    )

    spans = _extract_tool_content_spans(full_ids, tokenizer, tool_call_names)
    print(f"\nspans (count={len(spans)}):")
    for start, end, name in spans:
        decoded = tokenizer.decode(full_ids[start:end])
        print(f"  [{start:>4}..{end:>4}) name={name!r:>14} content={decoded!r}")

    assert len(spans) == 2, f"expected 2 spans, got {len(spans)}"

    s0_start, s0_end, s0_name = spans[0]
    s1_start, s1_end, s1_name = spans[1]
    assert s0_name == "run_code", f"first span should be run_code, got {s0_name}"
    assert s1_name == "lookup_doc", f"second span should be lookup_doc, got {s1_name}"

    # Verify the content tokens decode to the intended payloads (ignoring
    # any trailing newline that may or may not BPE-fuse into the last token).
    content0 = tokenizer.decode(full_ids[s0_start:s0_end]).strip()
    content1 = tokenizer.decode(full_ids[s1_start:s1_end]).strip()
    assert content0 == "stack: [4]", f"run_code content should be 'stack: [4]', got {content0!r}"
    assert content1 == "duplicates the top of stack", (
        f"lookup_doc content should be 'duplicates the top of stack', got {content1!r}"
    )

    # Verify the spans do NOT include the envelope tokens (<|im_start|>,
    # <tool_response>, etc.). Decode a slightly wider window and check the
    # outer tokens are envelope-y.
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tool_response_id = tokenizer.convert_tokens_to_ids("<tool_response>")
    tool_response_close_id = tokenizer.convert_tokens_to_ids("</tool_response>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for start, end, name in spans:
        for inside in full_ids[start:end]:
            assert inside not in (im_start_id, tool_response_id, tool_response_close_id, im_end_id), (
                f"span {name} content included envelope token {inside}"
            )

    print("\n[OK]  _extract_tool_content_spans behaves as specified.")


if __name__ == "__main__":
    main()
