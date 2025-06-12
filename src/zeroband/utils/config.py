from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Extract config file paths from CLI to pass to pydantic-settings as toml source
# This enables the use of `@` to pass config file paths to the CLI
def extract_toml_paths(args: list[str]) -> tuple[list[str], list[str]]:
    toml_paths = []
    remaining_args = args.copy()
    for arg, next_arg in zip(args, args[1:] + [""]):
        if arg.startswith("@"):
            if arg == "@":  # We assume that the next argument is a toml file path
                toml_paths.append(next_arg)
                remaining_args.remove(arg)
                remaining_args.remove(next_arg)
            else:  # We assume that the argument is a toml file path
                toml_paths.append(arg.replace("@", ""))
                remaining_args.remove(arg)
    return toml_paths, remaining_args


def to_kebab_case(args: list[str]) -> list[str]:
    """
    Converts CLI argument keys from snake case to kebab case.

    For example, `--max_batch_size 1` will be transformed `--max-batch-size 1`.
    """
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            args[i] = arg.replace("_", "-")
    return args
