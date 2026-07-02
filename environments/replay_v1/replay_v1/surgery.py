"""Pure functions over saved rollout records (WireTrace jsonl lines, as raw dicts).

A saved trace's ``nodes`` list is a message *forest*: parent pointers form trees, the main
conversation is rooted at node 0, and subagent conversations (or bare probe calls) get their
own roots appended to the same list, interleaved by index. Compaction leaves no marker — it
is a fork in the main tree whose new child is a non-sampled user message (the summary the
harness restarted from) with sampled turns beneath it. Everything here works on the raw
dicts so the buffer scan never pays pydantic validation for megabyte-sized lines.
"""

from collections import defaultdict
from typing import Any

Node = dict[str, Any]


def usable(record: dict) -> bool:
    """Whether a saved rollout is replayable at all. Saved step files contain the full
    arrival window: errored rollouts and synthesized cancel markers (empty ``nodes``,
    ``errors`` set) ride along and must be screened out."""
    return bool(record.get("nodes")) and not record.get("errors") and any(n["sampled"] for n in record["nodes"])


def is_replay_derived(task: dict) -> bool:
    """Whether a saved task dict came from a replay env (it carries the replay lineage
    keys — kept tight so an unrelated taskset with a `kind` field isn't silently
    excluded from default replay). Values mirror taskset.ReplayKind."""
    return task.get("kind") in ("continue", "recheck", "judge") and isinstance(task.get("source_task"), dict)


def unwrap_source_task(task: dict) -> dict:
    """The innermost original task under any chain of replay derivations — what inner-taskset
    scoring, tools, and container provisioning must be keyed on, however deep the chain."""
    while is_replay_derived(task) and task["source_task"]:
        task = task["source_task"]
    return task


def build_children(nodes: list[Node]) -> tuple[dict[int, list[int]], list[int]]:
    """Child lists (in node-index order, i.e. creation order) and roots of the forest."""
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for i, node in enumerate(nodes):
        parent = node["parent"]
        if parent is None:
            roots.append(i)
        else:
            children[parent].append(i)
    return children, roots


def main_tree(children: dict[int, list[int]]) -> set[int]:
    """Node indices of the main conversation: the component rooted at node 0. Subagent
    trees have their own roots and must not leak into seeds or transcripts."""
    seen: set[int] = set()
    stack = [0]
    while stack:
        i = stack.pop()
        seen.add(i)
        stack.extend(children.get(i, []))
    return seen


def _subtree_has_sampled(nodes: list[Node], children: dict[int, list[int]], start: int) -> bool:
    stack = [start]
    while stack:
        i = stack.pop()
        if nodes[i]["sampled"]:
            return True
        stack.extend(children.get(i, []))
    return False


def compaction_forks(nodes: list[Node], children: dict[int, list[int]], tree: set[int]) -> list[int]:
    """Compaction points in the main tree, as the post-compaction child node indices.

    A fork child is a compaction iff it is a non-sampled user message whose subtree
    contains sampled turns — the harness rewrote its prompt to a summary and kept going.
    The detector is structural, not marker-based; other fork kinds (duplicated tool
    results re-appended after retries, retried assistant twins) don't match it."""
    forks: list[int] = []
    for parent, siblings in children.items():
        if len(siblings) < 2 or parent not in tree:
            continue
        for child in siblings[1:]:
            message = nodes[child]["message"]
            if (
                message["role"] == "user"
                and not nodes[child]["sampled"]
                and _subtree_has_sampled(nodes, children, child)
            ):
                forks.append(child)
    return sorted(forks)


def path_to_root(nodes: list[Node], leaf: int) -> list[int]:
    path = []
    i: int | None = leaf
    while i is not None:
        path.append(i)
        i = nodes[i]["parent"]
    return path[::-1]


def final_leaf(children: dict[int, list[int]], tree: set[int]) -> int:
    """The last-written leaf of the main tree — the conversation's final state. The
    global max-index leaf may sit in a subagent tree, so restrict to the main tree."""
    return max(i for i in tree if i not in children)


def path_messages(nodes: list[Node], path: list[int]) -> list[dict]:
    return [nodes[i]["message"] for i in path]


def continue_seed(nodes: list[Node], fork_child: int) -> list[dict]:
    """CONTINUE seed: messages from the root down to (and including) the post-compaction
    user message — the exact prompt the original harness restarted from."""
    return path_messages(nodes, path_to_root(nodes, fork_child))


def recheck_seed(nodes: list[Node], children: dict[int, list[int]], tree: set[int], instruction: str) -> list[dict]:
    """RECHECK seed: the final branch's messages plus an appended check-your-work user turn.

    Rollouts routinely end truncated mid-tool-call (timeouts, context length); a trailing
    assistant's pending ``tool_calls`` must be stripped or the seed is API-malformed, and
    if that leaves it with no content it is dropped entirely."""
    path = path_to_root(nodes, final_leaf(children, tree))
    messages = [dict(m) for m in path_messages(nodes, path)]
    if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls"):
        messages[-1]["tool_calls"] = None
        if not messages[-1].get("content"):
            messages.pop()
    messages.append({"role": "user", "content": instruction})
    return messages


def content_text(content: Any) -> str:
    """Flatten message content (a string, or a list of content parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                parts.append(f"[{part.get('type', 'attachment')}]")
        return "\n".join(parts)
    return ""


def _render_message(message: dict, max_message_chars: int) -> str:
    text = content_text(message.get("content"))
    if len(text) > max_message_chars:
        text = f"{text[:max_message_chars]}\n[... truncated, {len(text)} chars total]"
    lines = [f"[{message['role'].upper()}]"]
    if text:
        lines.append(text)
    for call in message.get("tool_calls") or []:
        # Saved tool calls are flat {id, name, arguments} dicts, not OpenAI-nested.
        arguments = call.get("arguments") or ""
        if len(arguments) > 500:
            arguments = f"{arguments[:500]}...[truncated]"
        lines.append(f"[TOOL CALL] {call.get('name')}({arguments})")
    return "\n".join(lines)


def render_transcript(
    nodes: list[Node],
    children: dict[int, list[int]],
    tree: set[int],
    max_message_chars: int,
    max_total_chars: int,
) -> str:
    """JUDGE transcript: the final branch rendered as role-labeled text. Per-message
    truncation bounds pathological single messages (tool results reach 150K chars);
    a total budget with middle elision (keep the head, then as many trailing messages
    as fit) bounds the whole transcript."""
    path = path_to_root(nodes, final_leaf(children, tree))
    blocks = [_render_message(m, max_message_chars) for m in path_messages(nodes, path)]
    total = sum(len(b) + 2 for b in blocks)
    if total <= max_total_chars:
        return "\n\n".join(blocks)

    head = blocks[:2]  # system + opening user message: the task statement
    budget = max_total_chars - sum(len(b) + 2 for b in head)
    tail: list[str] = []
    for block in reversed(blocks[2:]):
        if budget - (len(block) + 2) < 0:
            break
        budget -= len(block) + 2
        tail.append(block)
    tail.reverse()
    elided = len(blocks) - len(head) - len(tail)
    return "\n\n".join([*head, f"[... {elided} messages elided ...]", *tail])
