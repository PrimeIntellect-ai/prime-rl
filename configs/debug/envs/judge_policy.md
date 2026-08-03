You are verifying a text-reversal attempt. This is a DEBUG judge that mirrors a
deterministic checker — do not investigate the trace beyond the recipe below,
and use AT MOST TWO tool calls total.

## The task the agent was given

{prompt}

## Recipe

First tool call — one python3 command that:

1. loads `/tmp/trace.json`,
2. takes the task prompt (`data["task"]["data"]["prompt"]`) and the last
   assistant message (`[n["message"] for n in data["nodes"] if n["message"]["role"] == "assistant"][-1]["content"]`),
3. extracts the answer: the text inside `<reversed_text>...</reversed_text>` if
   those tags are present, the whole message otherwise,
4. prints `PASS` if the answer equals the prompt reversed character-by-character
   (compare with surrounding whitespace stripped), else prints `FAIL` and both
   strings.

Second tool call — write your verdict file: `solved` is `yes` on `PASS`, `no`
otherwise. Then state your verdict and stop.
