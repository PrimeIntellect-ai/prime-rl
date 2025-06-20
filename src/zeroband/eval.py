# Import environment before any other imports
# ruff: noqa: I001
# from zeroband.eval import envs

from zeroband.utils.pydantic_config import parse_argv
from zeroband.eval.config import Config as EvalConfig
from zeroband.utils.utils import clean_exit


@clean_exit
def main(config: EvalConfig):
    # Initialize the logger
    print(config)


if __name__ == "__main__":
    main(parse_argv(EvalConfig))
