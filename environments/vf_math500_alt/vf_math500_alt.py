import verifiers as vf
from math_verify import parse, verify
from verifiers.parsers.parser import Parser
from verifiers.types import Messages
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, load_example_dataset


def vf_math_verify(parser: Parser, completion: Messages, answer: str, **kwargs) -> float:
    try:
        response = parser.parse_answer(completion) or ""
        if response == "":
            return 0.0
        if verify(
            parse(f"\\boxed{{{answer}}}", parsing_timeout=5),
            parse(response, parsing_timeout=5),
            timeout_seconds=5,
        ):
            return 1.0
        else:
            return 0.0
    except BaseException:
        return 0.0


def load_environment(system_prompt: str = BOXED_SYSTEM_PROMPT) -> vf.SingleTurnEnv:
    eval_dataset = load_example_dataset("math500")
    rubric = vf.Rubric(funcs=[vf_math_verify])

    vf_env = vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
    )
    return vf_env
