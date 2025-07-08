import sys
import warnings
from pathlib import Path
from typing import Annotated, ClassVar, Type, TypeVar

import tomli
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import (
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        """
        This allow to support setting None via toml files using the string "None"
        """
        if v == "None":
            return None
        return v


class BaseSettings(PydanticBaseSettings, BaseConfig):
    """
    Base settings class for all configs.
    """

    # These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
    _TOML_FILES: ClassVar[list[str]] = []

    toml_files: Annotated[
        list[str] | None,
        Field(
            description="List of extra TOML files to load (paths are relative to the TOML file containing this field). If provided, will override all other config files. Note: This field is only read from within configuration files - setting --toml-files from CLI has no effect.",
            exclude=True,
        ),
    ] = None

    @classmethod
    def set_toml_files(cls, toml_files: list[str]) -> None:
        """
        Set the global TOML files to be used for this config.
        These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
        """
        cls._TOML_FILES = toml_files

    @classmethod
    def clear_toml_files(cls) -> None:
        """
        Clear the global TOML files to be used for this config.
        """
        cls._TOML_FILES = []

    def set_unknown_args(self, unknown_args: list[str]) -> None:
        self._unknown_args = unknown_args

    def get_unknown_args(self) -> list[str]:
        return self._unknown_args

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["BaseSettings"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # This is a hacky way to dynamically load TOML file paths from CLI
        # https://github.com/pydantic/pydantic-settings/issues/259
        return (
            TomlConfigSettingsSource(settings_cls, toml_file=cls._TOML_FILES),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_prefix="PRIME_",
        env_nested_delimiter="__",
        cli_parse_args=False,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_use_class_docs_for_groups=True,
        nested_model_default_partial_update=True,  # Allows partial updates to nested model default objects
    )


def check_path_and_handle_inheritance(path: Path, seen_files: list[Path]) -> bool | None:
    """
    Recursively look for inheritance in a toml file. Return a list of all toml files to load.

    Example:
        If config.toml has `toml_files = ["base.toml"]` and base.toml has
        `toml_files = ["common.toml"]`, this returns ["config.toml", "base.toml", "common.toml"]

    Returns:
        True if some toml inheritance is detected, False otherwise.
    """
    if path in seen_files:
        return

    if not path.exists():
        raise FileNotFoundError(f"TOML file {path} does not exist")

    seen_files.append(path)

    with open(path, "rb") as f:
        data = tomli.load(f)

    recurence = False
    if "toml_files" in data:
        maybe_new_files = [path.parent / file for file in data["toml_files"]]

        files = [file for file in maybe_new_files if str(file).endswith(".toml")]
        # todo which should probably look for infinite inheritance loops here
        for file in files:
            recurence = True
            check_path_and_handle_inheritance(file, seen_files)

    return recurence


# Extract config file paths from CLI to pass to pydantic-settings as toml source
# This enables the use of `@` to pass config file paths to the CLI
def extract_toml_paths(args: list[str]) -> tuple[list[str], list[str]]:
    toml_paths = []
    remaining_args = args.copy()
    recurence = False
    cli_toml_file_count = 0
    for arg, next_arg in zip(args, args[1:] + [""]):
        if arg.startswith("@"):
            toml_path: str
            if arg == "@":  # We assume that the next argument is a toml file path
                toml_path = next_arg
                remaining_args.remove(arg)
                remaining_args.remove(next_arg)
            else:  # We assume that the argument is a toml file path
                remaining_args.remove(arg)
                toml_path = arg.replace("@", "")

            recurence = recurence or check_path_and_handle_inheritance(Path(toml_path), toml_paths)
            cli_toml_file_count += 1

    if recurence and cli_toml_file_count > 1:
        warnings.warn(
            f"{len(toml_paths)} TOML files are added via CLI ({', '.join(toml_paths)}) and at least one of them links to another file. This is not supported yet. Please either compose multiple config files via directly CLI or specify a single file linking to multiple other files"
        )

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


def get_all_fields(model: BaseModel | type) -> list[str]:
    if isinstance(model, BaseModel):
        model_cls = model.__class__
    else:
        model_cls = model

    fields = []
    for name, field in model_cls.model_fields.items():
        field_type = field.annotation
        fields.append(name)
        if hasattr(field_type, "model_fields"):
            sub_fields = get_all_fields(field_type)
            fields.extend(f"{name}.{sub}" for sub in sub_fields)
    return fields


def parse_unknown_args(args: list[str], config_cls: type) -> tuple[list[str], list[str]]:
    known_fields = get_all_fields(config_cls)
    known_args = []
    unknown_args = []
    i = 0
    n = len(args)

    def get_is_key(arg: str) -> bool:
        return arg.startswith("--") or arg.startswith("-")

    while i < n:
        is_key = get_is_key(args[i])
        has_value = False if i >= n - 1 or get_is_key(args[i + 1]) else True
        if not is_key:
            i += 1
            continue
        if args[i].startswith("--"):
            key = args[i][2:]
        else:
            key = args[i][1:]
        if key in known_fields:
            known_args.append(args[i])
            if has_value:
                known_args.append(args[i + 1])
                i += 2
            else:
                i += 1
        else:
            unknown_args.append(args[i])
            if has_value:
                unknown_args.append(args[i + 1])
                i += 2
            else:
                i += 1

    return known_args, unknown_args


T = TypeVar("T", bound=BaseSettings)


# Class[BaseSettings]
def parse_argv(config_cls: Type[T], allow_extras: bool = False) -> T:
    """
    Parse CLI arguments and TOML configuration files into a pydantic settings instance.

    Supports loading TOML files via @ syntax (e.g., @config.toml or @ config.toml).
    Automatically converts snake_case CLI args to kebab-case for pydantic compatibility.
    TOML files can inherit from other TOML files via the 'toml_files' field.

    Args:
        config_cls: A pydantic BaseSettings class to instantiate with parsed configuration.

    Returns:
        An instance of config_cls populated with values from TOML files and CLI args.
        CLI args take precedence over TOML file values.
    """
    toml_paths, cli_args = extract_toml_paths(sys.argv[1:])
    config_cls.set_toml_files(toml_paths)
    if allow_extras:
        cli_args, unknown_args = parse_unknown_args(cli_args, config_cls)
    config = config_cls(_cli_parse_args=to_kebab_case(cli_args))
    config_cls.clear_toml_files()
    if allow_extras:
        config.set_unknown_args(unknown_args)
    return config
