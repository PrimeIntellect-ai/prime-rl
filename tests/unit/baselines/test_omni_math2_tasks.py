import asyncio

from verifiers.utils.data_utils import extract_boxed_answer

from environments.omni_math2_singleturn.omni_math2_singleturn import (
    OMNI_MATH2_JUDGE_NEGATIVE_LABEL,
    OMNI_MATH2_JUDGE_POSITIVE_LABEL,
    OMNI_MATH2_JUDGE_PROMPT,
    OMNI_MATH2_JUDGE_SYSTEM_PROMPT,
    OMNI_MATH2_JUDGE_VARIANT_ID,
    PARSED_ANSWER_STATE_KEY,
    MathVerifyThenJudgeRubric,
    extract_omni_math2_answer,
    load_environment,
)
from scripts.evals.make_omni_math2_tasks import (
    allocate_largest_remainder,
    stratified_sample_indices,
    stratified_sample_plan,
)


class _Parser:
    def parse_answer(self, completion):
        return "parsed"


class _BoxedParser:
    def parse_answer(self, completion):
        content = completion[-1]["content"]
        return extract_boxed_answer(content, strict=True)


class _OmniParser:
    def parse_answer(self, completion):
        content = completion[-1]["content"]
        return extract_omni_math2_answer(content)


class _CountingParser:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def parse_answer(self, completion):
        self.calls += 1
        return self.answer


class _MathRubric:
    async def correct_answer(self, **kwargs):
        return 0.0

    async def cleanup(self, state):
        pass

    async def teardown(self):
        pass


class _JudgeRubric:
    parser = _Parser()
    judge_positive_label = "CORRECT"

    def __init__(self):
        self.calls = []
        self.parsed_responses = []

    async def judge(self, prompt, completion, answer, state):
        self.calls.append((prompt, completion, answer))
        self.parsed_responses.append(self.parser.parse_answer(completion))
        state["judge_decision_last"] = {"reward": 1.0}
        state["judge_response"] = {"fake": "CORRECT"}
        return "CORRECT"

    async def cleanup(self, state):
        pass

    async def teardown(self):
        pass


class _JudgeClient:
    pass


def test_allocate_largest_remainder_respects_capacity_and_total():
    counts = {"a": 10, "b": 3, "c": 1}

    allocation = allocate_largest_remainder(counts, sample_size=6)

    assert sum(allocation.values()) == 6
    assert all(0 <= allocation[key] <= counts[key] for key in counts)
    assert allocation["a"] >= allocation["b"] >= allocation["c"]


def test_stratified_sample_indices_is_deterministic_without_duplicates():
    rows = [
        {"stratum": "x", "id": 1},
        {"stratum": "x", "id": 2},
        {"stratum": "x", "id": 3},
        {"stratum": "y", "id": 4},
        {"stratum": "y", "id": 5},
        {"stratum": "z", "id": 6},
    ]

    first = stratified_sample_indices(
        rows,
        sample_size=4,
        seed=17,
        stratum_key=lambda row: row["stratum"],
    )
    second = stratified_sample_indices(
        rows,
        sample_size=4,
        seed=17,
        stratum_key=lambda row: row["stratum"],
    )

    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
    selected_strata = [rows[index]["stratum"] for index in first]
    assert selected_strata.count("x") == 2
    assert selected_strata.count("y") == 1
    assert selected_strata.count("z") == 1


def test_stratified_sample_indices_uses_seed_for_tied_singleton_strata():
    rows = [{"stratum": str(index)} for index in range(20)]

    first = stratified_sample_indices(
        rows,
        sample_size=5,
        seed=1,
        stratum_key=lambda row: row["stratum"],
    )
    second = stratified_sample_indices(
        rows,
        sample_size=5,
        seed=2,
        stratum_key=lambda row: row["stratum"],
    )

    assert first != second
    assert len(first) == len(second) == 5


def test_stratified_sample_plan_reports_allocation_and_quotas():
    rows = [{"stratum": "x"} for _ in range(3)] + [{"stratum": "y"}]

    indices, allocation, quotas = stratified_sample_plan(
        rows,
        sample_size=3,
        seed=7,
        stratum_key=lambda row: row["stratum"],
    )

    assert len(indices) == 3
    assert sum(allocation.values()) == 3
    assert allocation == {"x": 2, "y": 1}
    assert quotas == {"x": 2.25, "y": 0.75}


def test_hybrid_rubric_judge_fallback_calls_judge_directly():
    async def run():
        judge_rubric = _JudgeRubric()
        parser = _Parser()
        rubric = MathVerifyThenJudgeRubric(
            parser=parser,
            math_rubric=_MathRubric(),
            judge_rubric=judge_rubric,
        )
        prompt = [{"role": "user", "content": "Question?"}]
        completion = [{"role": "assistant", "content": "\\boxed{wrong}"}]
        state = {"math_verify_score": 0.0}

        score = await rubric.judge_score(
            prompt=prompt,
            completion=completion,
            answer="right",
            state=state,
        )

        assert score == 1.0
        assert judge_rubric.parser.base_parser is parser
        assert judge_rubric.parsed_responses == ["parsed"]
        assert judge_rubric.calls == [(prompt, completion, "right")]
        assert state["judge_score"] == 1.0
        assert state["judge_response"] == {"fake": "CORRECT"}

    asyncio.run(run())


def test_omni_math2_env_uses_task_aware_judge_prompt_by_default():
    from datasets import Dataset

    env = load_environment(
        dataset=Dataset.from_list(
            [
                {
                    "problem": "Which graph is horizontal? (A) Graph P (B) Graph Q",
                    "answer": "Graph Q",
                    "id": 1,
                }
            ]
        ),
        judge_client=_JudgeClient(),
        judge_base_url=None,
        prompts_ref="configs/baselines/omni_math2_prompt_pack.yaml",
    )
    rubric = next(rubric for rubric in env.rubric.rubrics if isinstance(rubric, MathVerifyThenJudgeRubric))

    assert rubric.judge_rubric.judge_system_prompt == OMNI_MATH2_JUDGE_SYSTEM_PROMPT
    assert rubric.judge_rubric.judge_prompt == OMNI_MATH2_JUDGE_PROMPT
    assert rubric.judge_rubric.judge_positive_label == OMNI_MATH2_JUDGE_POSITIVE_LABEL
    assert rubric.judge_rubric.judge_negative_label == OMNI_MATH2_JUDGE_NEGATIVE_LABEL
    assert rubric.judge_rubric.judge_variant_id == OMNI_MATH2_JUDGE_VARIANT_ID
    assert "Do not solve the problem yourself" in rubric.judge_rubric.judge_system_prompt
    assert "map that letter to the option text" in rubric.judge_rubric.judge_system_prompt


def test_hybrid_rubric_accepts_inline_choice_label_alias():
    async def run():
        judge_rubric = _JudgeRubric()
        rubric = MathVerifyThenJudgeRubric(
            parser=_BoxedParser(),
            math_rubric=_MathRubric(),
            judge_rubric=judge_rubric,
        )
        prompt = [
            {
                "role": "user",
                "content": (
                    "Which expression is not equivalent to $3x + 6$? "
                    "(A) $3(x + 2)$ "
                    "(B) $\\frac{-9x - 18}{-3}$ "
                    "(C) $\\frac{1}{3}(3x) + \\frac{2}{3}(9)$ "
                    "(D) $\\frac{1}{3}(9x + 18)$\n\n"
                    "Work through the problem before giving the answer."
                ),
            }
        ]
        completion = [{"role": "assistant", "content": "Final answer: \\boxed{(C)}"}]
        state = {"math_verify_score": 0.0}

        alias_score = await rubric.choice_alias_score(
            prompt=prompt,
            completion=completion,
            answer="\\frac{1}{3}(3x) + \\frac{2}{3}(9)",
            state=state,
        )
        judge_score = await rubric.judge_score(
            prompt=prompt,
            completion=completion,
            answer="\\frac{1}{3}(3x) + \\frac{2}{3}(9)",
            state=state,
        )
        final_score = await rubric.correct_answer(state=state)

        assert alias_score == 1.0
        assert judge_score == 1.0
        assert final_score == 1.0
        assert state["choice_aliases"] == ["C"]
        assert judge_rubric.calls == []

    asyncio.run(run())


def test_hybrid_rubric_rejects_non_matching_inline_choice_label_alias():
    async def run():
        rubric = MathVerifyThenJudgeRubric(
            parser=_BoxedParser(),
            math_rubric=_MathRubric(),
            judge_rubric=None,
        )
        prompt = [
            {
                "role": "user",
                "content": (
                    "Which expression is not equivalent to $3x + 6$? "
                    "(A) $3(x + 2)$ "
                    "(B) $\\frac{-9x - 18}{-3}$ "
                    "(C) $\\frac{1}{3}(3x) + \\frac{2}{3}(9)$ "
                    "(D) $\\frac{1}{3}(9x + 18)$"
                ),
            }
        ]
        completion = [{"role": "assistant", "content": "Final answer: \\boxed{D}"}]
        state = {"math_verify_score": 0.0}

        score = await rubric.choice_alias_score(
            prompt=prompt,
            completion=completion,
            answer="\\frac{1}{3}(3x) + \\frac{2}{3}(9)",
            state=state,
        )

        assert score == 0.0
        assert state["choice_aliases"] == ["C"]

    asyncio.run(run())


def test_hybrid_rubric_does_not_treat_plain_symbolic_answer_as_choice():
    async def run():
        rubric = MathVerifyThenJudgeRubric(
            parser=_BoxedParser(),
            math_rubric=_MathRubric(),
            judge_rubric=None,
        )
        prompt = [{"role": "user", "content": "Find the value of C."}]
        completion = [{"role": "assistant", "content": "Final answer: \\boxed{C}"}]
        state = {"math_verify_score": 0.0}

        score = await rubric.choice_alias_score(
            prompt=prompt,
            completion=completion,
            answer="C",
            state=state,
        )

        assert score == 0.0
        assert state["choice_aliases"] == []

    asyncio.run(run())


def test_hybrid_rubric_accepts_targeted_text_aliases_without_judge():
    async def run():
        judge_rubric = _JudgeRubric()
        rubric = MathVerifyThenJudgeRubric(
            parser=_BoxedParser(),
            math_rubric=_MathRubric(),
            judge_rubric=judge_rubric,
        )
        prompt = [{"role": "user", "content": "Find all natural n."}]
        cases = [
            (
                [{"role": "assistant", "content": "Final answer: \\boxed{\\emptyset}"}],
                "\\text{No such } n \\text{ exists}",
            ),
            (
                [{"role": "assistant", "content": "Final answer: \\boxed{finite}"}],
                "The set of positive integers \\( n \\) such that property P holds is finite.",
            ),
        ]

        for completion, answer in cases:
            state = {"math_verify_score": 0.0, "choice_alias_score": 0.0}
            text_score = await rubric.text_alias_score(
                completion=completion,
                answer=answer,
                state=state,
            )
            judge_score = await rubric.judge_score(
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )

            assert text_score == 1.0
            assert judge_score == 1.0

        assert judge_rubric.calls == []

    asyncio.run(run())


def test_hybrid_rubric_caches_parsed_answer_for_alias_checks():
    async def run():
        parser = _CountingParser("\\emptyset")
        rubric = MathVerifyThenJudgeRubric(
            parser=parser,
            math_rubric=_MathRubric(),
            judge_rubric=None,
        )
        completion = [{"role": "assistant", "content": "Final answer: \\boxed{\\emptyset}"}]
        state = {"math_verify_score": 0.0}

        choice_score = await rubric.choice_alias_score(
            prompt=[{"role": "user", "content": "Find all natural n."}],
            completion=completion,
            answer="\\text{No such } n \\text{ exists}",
            state=state,
        )
        text_score = await rubric.text_alias_score(
            completion=completion,
            answer="\\text{No such } n \\text{ exists}",
            state=state,
        )

        assert choice_score == 0.0
        assert text_score == 1.0
        assert parser.calls == 1
        assert state[PARSED_ANSWER_STATE_KEY] == "\\emptyset"

    asyncio.run(run())


def test_hybrid_rubric_reuses_cached_parse_for_judge_fallback():
    async def run():
        parser = _CountingParser("wrong")
        judge_rubric = _JudgeRubric()
        rubric = MathVerifyThenJudgeRubric(
            parser=parser,
            math_rubric=_MathRubric(),
            judge_rubric=judge_rubric,
        )
        prompt = [{"role": "user", "content": "Compute the value."}]
        completion = [{"role": "assistant", "content": "Final answer: wrong"}]
        state = {"math_verify_score": 0.0}

        await rubric.choice_alias_score(
            prompt=prompt,
            completion=completion,
            answer="right",
            state=state,
        )
        await rubric.text_alias_score(
            completion=completion,
            answer="right",
            state=state,
        )
        score = await rubric.judge_score(
            prompt=prompt,
            completion=completion,
            answer="right",
            state=state,
        )

        assert score == 1.0
        assert parser.calls == 1
        assert judge_rubric.parsed_responses == ["wrong"]
        assert state[PARSED_ANSWER_STATE_KEY] == "wrong"

    asyncio.run(run())


def test_omni_math2_parser_accepts_final_answer_line_without_box():
    assert extract_omni_math2_answer("Reasoning...\nFinal answer: 7.") == "7"
    assert extract_omni_math2_answer("Reasoning. The answer is 90.\n</think>\n\n90") == "90"
    assert extract_omni_math2_answer("Thus the answer is no such n exists.") == "no such n"


def test_hybrid_rubric_judges_final_answer_line_without_box():
    async def run():
        judge_rubric = _JudgeRubric()
        rubric = MathVerifyThenJudgeRubric(
            parser=_OmniParser(),
            math_rubric=_MathRubric(),
            judge_rubric=judge_rubric,
        )
        prompt = [{"role": "user", "content": "Compute the value."}]
        completion = [{"role": "assistant", "content": "Final answer: 7."}]
        state = {
            "math_verify_score": 0.0,
            "choice_alias_score": 0.0,
            "text_alias_score": 0.0,
        }

        score = await rubric.judge_score(
            prompt=prompt,
            completion=completion,
            answer="7",
            state=state,
        )

        assert score == 1.0
        assert state["judge_score"] == 1.0
        assert judge_rubric.parsed_responses == ["7"]

    asyncio.run(run())
