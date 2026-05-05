from __future__ import annotations

import re
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

import verifiers as vf
from datasets import Dataset
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.parsers.maybe_think_parser import MaybeThinkParser
from verifiers.types import ClientConfig
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.hf_tasks import (
    DEFAULT_JUDGE_PROMPT_PACK,
    JudgePromptKind,
    load_hf_dataset,
    make_rubric,
    normalize_hf_dataset,
)
from verifiers.utils.judge_prompts import normalize_verdict_token

INLINE_CHOICE_RE = re.compile(r"\(([A-Z])\)\s*")
TRAILING_PROMPT_RE = re.compile(r"\n\s*Work through the problem\b", re.IGNORECASE)
LATEX_TEXT_RE = re.compile(r"\\text\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(
    r"(?:(?:final\s+answer|answer)\s*:|(?:the\s+)?answer\s+is)\s*(?P<answer>[^\n\r<]+)",
    re.IGNORECASE,
)
NO_SOLUTION_RE = re.compile(
    r"\bno\s+such\s+(?:n|natural\s+number|natural\s+numbers|integer|integers)\b",
    re.IGNORECASE,
)
MAX_FALLBACK_ANSWER_CHARS = 200
PARSED_ANSWER_STATE_KEY = "omni_math2_parsed_answer"
PARSE_CACHE_STATE: ContextVar[vf.State | None] = ContextVar("omni_math2_parse_cache_state", default=None)


def _message_content(messages: vf.Messages) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            role = message.get("role")
        else:
            content = getattr(message, "content", "")
            role = getattr(message, "role", None)
        if role == "user" and isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _canonical_answer_text(text: str) -> str:
    text = text.strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    text = re.sub(r"\s+", "", text)
    return text.strip(".,;:")


def _inline_choice_aliases(question: str, answer: str) -> set[str]:
    question = TRAILING_PROMPT_RE.split(question, maxsplit=1)[0]
    matches = list(INLINE_CHOICE_RE.finditer(question))
    if len(matches) < 2:
        return set()

    labels = [match.group(1) for match in matches]
    expected = [chr(ord("A") + i) for i in range(len(labels))]
    if labels != expected:
        return set()

    target = _canonical_answer_text(answer)
    aliases: set[str] = set()
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        option = _canonical_answer_text(question[start:end])
        if option == target:
            aliases.add(match.group(1))
    return aliases


def _canonical_choice_response(response: str | None) -> str | None:
    if response is None:
        return None
    response = response.strip()
    text_match = re.fullmatch(r"\\text\{([^{}]+)\}", response)
    if text_match is not None:
        response = text_match.group(1).strip()
    match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", response.upper())
    return match.group(1) if match is not None else None


def _strip_answer_fence(text: str) -> str:
    text = text.strip()
    text = text.strip("`")
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    return text.strip().strip(".;,")


def extract_omni_math2_answer(text: str, *, strict: bool = True) -> str | None:
    boxed = extract_boxed_answer(text, strict=strict)
    if boxed:
        return boxed

    tail = text[-4000:]
    for match in reversed(list(FINAL_ANSWER_RE.finditer(tail))):
        answer = _strip_answer_fence(match.group("answer"))
        if NO_SOLUTION_RE.search(answer):
            return "no such n"
        boxed = extract_boxed_answer(answer, strict=strict)
        if boxed:
            return boxed
        if answer and len(answer) <= MAX_FALLBACK_ANSWER_CHARS:
            return answer

    if NO_SOLUTION_RE.search(tail):
        return "no such n"

    return None


def _plain_text_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = LATEX_TEXT_RE.sub(lambda match: match.group(1), text)
    text = text.replace("\\emptyset", "empty set")
    text = re.sub(r"\\[()]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _text_alias_matches(response: str | None, answer: str) -> bool:
    response_text = _plain_text_answer(response)
    answer_text = _plain_text_answer(answer)
    if not response_text or not answer_text:
        return False

    empty_set_aliases = {
        "empty set",
        "none",
        "no solution",
        "no solutions",
        "no such n",
        "no such natural number",
        "no such natural numbers",
    }
    if "no such" in answer_text and response_text in empty_set_aliases:
        return True

    if answer_text.endswith(" is finite") and response_text == "finite":
        return True

    return False


def _validated_parsed_answer(parsed: Any) -> str | None:
    if parsed is None or isinstance(parsed, str):
        return parsed
    raise TypeError(f"Expected parsed answer to be str or None, got {type(parsed).__name__}")


class _StateCachedParser:
    def __init__(self, parser: vf.Parser):
        self.base_parser = parser

    def parse_answer(self, completion: vf.Messages) -> str | None:
        state = PARSE_CACHE_STATE.get()
        if state is None:
            return _validated_parsed_answer(self.base_parser.parse_answer(completion))
        if PARSED_ANSWER_STATE_KEY not in state:
            state[PARSED_ANSWER_STATE_KEY] = self.base_parser.parse_answer(completion)
        return _validated_parsed_answer(state[PARSED_ANSWER_STATE_KEY])


class MathVerifyThenJudgeRubric(vf.Rubric):
    def __init__(
        self,
        *,
        parser: vf.Parser,
        math_rubric: vf.MathRubric,
        judge_rubric: vf.Rubric | None,
    ):
        super().__init__(parser=parser)
        self.math_rubric = math_rubric
        self.judge_rubric = judge_rubric
        if judge_rubric is not None:
            judge_rubric.parser = _StateCachedParser(parser)
        self.add_reward_func(self.math_verify_score, weight=0)
        self.add_reward_func(self.choice_alias_score, weight=0)
        self.add_reward_func(self.text_alias_score, weight=0)
        if judge_rubric is not None:
            self.add_reward_func(self.judge_score, weight=0)
        self.add_reward_func(self.correct_answer, weight=1)

    def _parsed_answer(self, completion: vf.Messages, state: vf.State) -> str | None:
        if PARSED_ANSWER_STATE_KEY not in state:
            state[PARSED_ANSWER_STATE_KEY] = self.parser.parse_answer(completion)
        return _validated_parsed_answer(state[PARSED_ANSWER_STATE_KEY])

    async def math_verify_score(
        self,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        score = await self.math_rubric.correct_answer(
            parser=self.parser,
            completion=completion,
            answer=answer,
        )
        state["math_verify_score"] = score
        return score

    async def choice_alias_score(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if state.get("math_verify_score", 0.0) == 1.0:
            score = 1.0
            state["choice_alias_score"] = score
            return score

        aliases = _inline_choice_aliases(_message_content(prompt), answer)
        response = _canonical_choice_response(self._parsed_answer(completion, state))
        score = 1.0 if response is not None and response in aliases else 0.0
        state["choice_aliases"] = sorted(aliases)
        state["choice_alias_score"] = score
        return score

    async def text_alias_score(
        self,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        response = self._parsed_answer(completion, state)
        score = 1.0 if _text_alias_matches(response, answer) else 0.0
        state["text_alias_score"] = score
        return score

    async def judge_score(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if state.get("math_verify_score", 0.0) == 1.0:
            score = float(state.get("math_verify_score", 0.0))
            state["judge_score"] = score
            return score

        if state.get("choice_alias_score", 0.0) == 1.0:
            score = float(state.get("choice_alias_score", 0.0))
            state["judge_score"] = score
            return score

        if state.get("text_alias_score", 0.0) == 1.0:
            score = float(state.get("text_alias_score", 0.0))
            state["judge_score"] = score
            return score

        parsed = self._parsed_answer(completion, state)
        if parsed is None or not parsed.strip():
            state["judge_score"] = 0.0
            return 0.0

        token = PARSE_CACHE_STATE.set(state)
        try:
            raw_grade = await self.judge_rubric.judge(prompt, completion, answer, state)
        finally:
            PARSE_CACHE_STATE.reset(token)
        decision = state.get("judge_decision_last")
        if isinstance(decision, dict) and isinstance(decision.get("reward"), int | float):
            score = float(decision["reward"])
        else:
            positive_label = getattr(
                self.judge_rubric,
                "judge_positive_label",
                "CORRECT",
            )
            score = 1.0 if normalize_verdict_token(raw_grade) == positive_label else 0.0

        state["judge_score"] = score
        return score

    async def correct_answer(self, state: vf.State, **kwargs: Any) -> float:
        return float(
            state.get("math_verify_score", 0.0)
            or state.get("choice_alias_score", 0.0)
            or state.get("text_alias_score", 0.0)
            or state.get("judge_score", 0.0)
        )

    async def cleanup(self, state: vf.State) -> None:
        await self.math_rubric.cleanup(state)
        if self.judge_rubric is not None:
            await self.judge_rubric.cleanup(state)
        await super().cleanup(state)

    async def teardown(self) -> None:
        await self.math_rubric.teardown()
        if self.judge_rubric is not None:
            await self.judge_rubric.teardown()
        await super().teardown()


def load_environment(
    dataset_name: str = "json",
    dataset_subset: str | None = None,
    dataset_split: str = "train",
    eval_dataset_split: str | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    dataset_streaming: bool = False,
    dataset_columns: Sequence[str] | None = None,
    dataset_streaming_shuffle_buffer_size: int | None = None,
    dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    question_key: str = "problem",
    answer_key: str = "answer",
    example_id_key: str | None = "id",
    task_name: str = "omni_math2",
    info_keys: Sequence[str] | None = None,
    include_raw_info: bool = False,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    prompts_ref: str | None = None,
    prompt_role: str = "debater_a",
    prompt_phase: str = "propose",
    seed: int = 0,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    use_judge_fallback: bool = True,
    judge_client: Any | None = None,
    judge_model: str = "gpt-5.4-mini",
    judge_base_url: str | None = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    judge_sampling_args: dict[str, Any] | None = None,
    judge_prompt_pack: str | None = DEFAULT_JUDGE_PROMPT_PACK,
    judge_prompt_kind: JudgePromptKind = "grader",
    judge_system_prompt: str | None = None,
    judge_prompt: str | None = None,
    judge_positive_label: str | None = None,
    judge_negative_label: str | None = None,
    judge_cache_enabled: bool = True,
    judge_cache_size: int = 50000,
    judge_max_retries: int = 2,
    judge_retry_delay_s: float = 0.5,
    judge_persistent_cache_path: str | None = None,
    judge_persistent_cache_min_samples: int = 1,
    judge_rubric_family: str = "omni_math2_hybrid_math_v1",
    judge_variant_id: str = "hybrid_math_reference_anchored_v1",
    math_verify_timeout_seconds: float = 5,
    math_verify_max_workers: int = 50,
    **extra: Any,
) -> vf.Environment:
    parser = MaybeThinkParser(extract_fn=extract_omni_math2_answer)
    if use_judge_fallback and judge_client is None and judge_base_url is not None:
        judge_client = OpenAIChatCompletionsClient(
            ClientConfig(
                api_base_url=judge_base_url,
                api_key_var=judge_api_key_var,
            )
        )

    math_rubric = vf.MathRubric(
        parser=parser,
        timeout_seconds=math_verify_timeout_seconds,
        max_workers=math_verify_max_workers,
    )
    judge_rubric = None
    if use_judge_fallback:
        judge_rubric = make_rubric(
            task_type="open_ended",
            judge_client=judge_client,
            judge_model=judge_model,
            judge_base_url=None if judge_client is not None else judge_base_url,
            judge_api_key_var=judge_api_key_var,
            judge_sampling_args=judge_sampling_args or {"temperature": 0.0, "max_completion_tokens": 64},
            judge_prompt_pack=judge_prompt_pack,
            judge_prompt_kind=judge_prompt_kind,
            **({"judge_system_prompt": judge_system_prompt} if judge_system_prompt is not None else {}),
            **({"judge_prompt": judge_prompt} if judge_prompt is not None else {}),
            **({"judge_positive_label": judge_positive_label} if judge_positive_label is not None else {}),
            **({"judge_negative_label": judge_negative_label} if judge_negative_label is not None else {}),
            judge_cache_enabled=judge_cache_enabled,
            judge_cache_size=judge_cache_size,
            judge_max_retries=judge_max_retries,
            judge_retry_delay_s=judge_retry_delay_s,
            judge_persistent_cache_path=judge_persistent_cache_path,
            judge_persistent_cache_min_samples=judge_persistent_cache_min_samples,
            judge_rubric_family=judge_rubric_family,
            judge_variant_id=judge_variant_id,
        )

    rubric = MathVerifyThenJudgeRubric(
        parser=parser,
        math_rubric=math_rubric,
        judge_rubric=judge_rubric,
    )

    def normalize_math_dataset(raw: Dataset, limit: int, shuffle_seed: int) -> Dataset:
        return normalize_hf_dataset(
            raw,
            task_type="math",
            question_key=question_key,
            answer_key=answer_key,
            example_id_key=example_id_key,
            task_name=task_name,
            info_keys=info_keys,
            include_raw_info=include_raw_info,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            prompts_ref=prompts_ref,
            prompt_role=prompt_role,
            prompt_phase=prompt_phase,
            seed=shuffle_seed,
            num_examples=limit,
        )

    def build_dataset(split: str, limit: int, shuffle_seed: int) -> Dataset:
        if dataset is not None:
            raw = dataset
        else:
            raw = load_hf_dataset(
                dataset_name,
                dataset_subset=dataset_subset,
                dataset_split=split,
                data_files=data_files,
                streaming=dataset_streaming,
                columns=dataset_columns,
                streaming_limit=limit,
                streaming_shuffle_buffer_size=dataset_streaming_shuffle_buffer_size,
                streaming_seed=shuffle_seed,
            )
        return normalize_math_dataset(raw, limit, shuffle_seed)

    def build_eval_dataset() -> Dataset:
        if eval_dataset is not None:
            return normalize_math_dataset(eval_dataset, num_eval_examples, seed + 1)
        return build_dataset(eval_dataset_split or dataset_split, num_eval_examples, seed + 1)

    return vf.SingleTurnEnv(
        dataset=lambda: build_dataset(dataset_split, num_train_examples, seed),
        eval_dataset=build_eval_dataset,
        parser=parser,
        rubric=rubric,
        **extra,
    )
