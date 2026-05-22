"""Generate docs/reference.md from the Pydantic config models.

Walks every top-level user-facing config (RLConfig, SFTConfig, TrainerConfig,
OrchestratorConfig, InferenceConfig), recursively renders its nested sub-configs
and discriminated unions, and writes a single Markdown reference page.

Run from the project root:
    uv run python scripts/generate_docs_reference.py
"""

from __future__ import annotations

import io
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_config.cli import _extract_field_docstrings

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import TrainerConfig

OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "reference.md"


@dataclass
class Entrypoint:
    slug: str
    title: str
    cls: type[BaseModel]
    blurb: str


ENTRYPOINTS = [
    Entrypoint(
        slug="rl",
        title="`rl` — Full RL training",
        cls=RLConfig,
        blurb=(
            "The `rl` entrypoint composes a trainer, orchestrator, and (optionally) inference server into a single "
            "co-located deployment. Sub-configs under `[trainer]`, `[orchestrator]`, and `[inference]` mirror the "
            "standalone entrypoints below, with shared knobs (model name, output dir, W&B run name, …) lifted to "
            "the top level so they only need to be set once."
        ),
    ),
    Entrypoint(
        slug="sft",
        title="`sft` — Supervised fine-tuning",
        cls=SFTConfig,
        blurb="The `sft` entrypoint runs supervised fine-tuning on a tokenized dataset.",
    ),
    Entrypoint(
        slug="trainer",
        title="`trainer` — Standalone trainer",
        cls=TrainerConfig,
        blurb=(
            "The `trainer` entrypoint runs only the trainer process. It expects rollouts to be shipped in via the "
            "configured transport (filesystem or ZMQ) by an external orchestrator."
        ),
    ),
    Entrypoint(
        slug="orchestrator",
        title="`orchestrator` — Standalone orchestrator",
        cls=OrchestratorConfig,
        blurb=(
            "The `orchestrator` entrypoint runs only the orchestrator process. It expects a separately-launched "
            "inference server to serve rollouts, and ships completed rollouts to a separately-launched trainer "
            "over the configured transport."
        ),
    ),
    Entrypoint(
        slug="inference",
        title="`inference` — Standalone vLLM server",
        cls=InferenceConfig,
        blurb=(
            "The `inference` entrypoint launches a vLLM server (or a disaggregated prefill/decode pair) that "
            "serves OpenAI-compatible completions to the orchestrator."
        ),
    ),
]


def is_pydantic_model(t: object) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def unwrap_annotated(t: object) -> object:
    while typing.get_origin(t) is typing.Annotated:
        t = typing.get_args(t)[0]
    return t


def discriminated_variants(field: FieldInfo) -> list[type[BaseModel]] | None:
    """Return the variant model classes if `field` is a discriminated union over BaseModels."""
    if field.discriminator is None:
        return None
    return _union_models(unwrap_annotated(field.annotation))


def _union_models(t: object) -> list[type[BaseModel]] | None:
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if all(is_pydantic_model(a) for a in args):
            return args
    return None


def fmt_type(annotation: object) -> str:
    """Render a type annotation as compact Markdown-safe text."""
    annotation = unwrap_annotated(annotation)
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(annotation)
        return " \\| ".join(fmt_type(a) for a in args)
    if origin is typing.Literal:
        return " \\| ".join(repr(a) for a in typing.get_args(annotation))
    if origin is list:
        return f"list[{fmt_type(typing.get_args(annotation)[0])}]"
    if origin is dict:
        k, v = typing.get_args(annotation)
        return f"dict[{fmt_type(k)}, {fmt_type(v)}]"
    if origin is tuple:
        return f"tuple[{', '.join(fmt_type(a) for a in typing.get_args(annotation))}]"
    if annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def fmt_default(field: FieldInfo) -> str:
    if field.is_required():
        return "*required*"
    default = field.default
    factory = field.default_factory
    if factory is not None:
        try:
            default = factory()
        except Exception:
            return "*factory*"
    if isinstance(default, BaseModel):
        return f"`{default.__class__.__name__}()`"
    if default is None:
        return "`None`"
    if isinstance(default, str):
        return f"`{default!r}`"
    if isinstance(default, Path):
        return f"`{str(default)!r}`"
    if isinstance(default, (list, dict)) and not default:
        return f"`{default!r}`"
    return f"`{default!r}`"


def fmt_constraints(field: FieldInfo) -> str:
    parts: list[str] = []
    for m in field.metadata:
        for attr, sym in (("ge", "≥"), ("gt", ">"), ("le", "≤"), ("lt", "<")):
            if hasattr(m, attr):
                v = getattr(m, attr)
                if v is not None:
                    parts.append(f"{sym}{v}")
        if hasattr(m, "min_length") and m.min_length is not None:
            parts.append(f"len ≥ {m.min_length}")
        if hasattr(m, "max_length") and m.max_length is not None:
            parts.append(f"len ≤ {m.max_length}")
    return ", ".join(parts)


def slug(parts: list[str]) -> str:
    return "-".join(parts).replace("_", "-").replace(".", "-").lower()


class Writer:
    def __init__(self) -> None:
        self.buf = io.StringIO()
        self.toc: list[tuple[int, str, str]] = []  # (level, label, anchor)

    def h(self, level: int, text: str, anchor: str | None = None) -> None:
        if anchor:
            self.buf.write(f'<a id="{anchor}"></a>\n')
            self.toc.append((level, text, anchor))
        self.buf.write(f"{'#' * level} {text}\n\n")

    def p(self, text: str) -> None:
        self.buf.write(f"{text}\n\n")

    def raw(self, text: str) -> None:
        self.buf.write(text)


def render_field_row(
    writer: Writer,
    path: str,
    field: FieldInfo,
    docstring: str,
) -> None:
    name = f"`{path}`"
    type_str = fmt_type(field.annotation)
    default = fmt_default(field)
    constraints = fmt_constraints(field)
    desc = (field.description or docstring or "").strip().replace("\n", " ")
    if constraints:
        desc = f"_{constraints}._ {desc}" if desc else f"_{constraints}._"
    writer.raw(f"| {name} | {type_str} | {default} | {desc} |\n")


def render_model(
    writer: Writer,
    model_cls: type[BaseModel],
    path_prefix: str,
    anchor_prefix: list[str],
    depth: int,
    seen: set[type[BaseModel]],
) -> None:
    """Render the fields of `model_cls` and recurse into nested BaseConfig sub-fields."""
    docstrings = _extract_field_docstrings(model_cls)
    nested: list[tuple[str, type[BaseModel], FieldInfo, str]] = []
    union_fields: list[tuple[str, list[type[BaseModel]], FieldInfo, str]] = []
    flat_fields: list[tuple[str, FieldInfo, str]] = []

    for name, field in model_cls.model_fields.items():
        full = f"{path_prefix}.{name}" if path_prefix else name
        ds = docstrings.get(name, "")
        ann = field.annotation
        variants = discriminated_variants(field)
        if variants is not None:
            union_fields.append((full, variants, field, ds))
            continue
        unwrapped = unwrap_annotated(ann)
        if is_pydantic_model(unwrapped):
            nested.append((full, unwrapped, field, ds))
            continue
        # Optional[BaseConfig] case (e.g. `LoRAConfig | None`)
        origin = typing.get_origin(unwrapped)
        if origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(unwrapped) if a is not type(None)]
            if len(args) == 1 and is_pydantic_model(args[0]):
                nested.append((full, args[0], field, ds))
                continue
        flat_fields.append((full, field, ds))

    if flat_fields:
        writer.raw("| Field | Type | Default | Description |\n")
        writer.raw("|---|---|---|---|\n")
        for full, field, ds in flat_fields:
            render_field_row(writer, full, field, ds)
        writer.raw("\n")

    for full, child_cls, field, ds in nested:
        sub_anchor = anchor_prefix + [full.split(".")[-1]]
        heading = f"`{full}`"
        writer.h(min(depth + 1, 6), heading, anchor=slug(sub_anchor))
        blurb = (field.description or ds or "").strip()
        if blurb:
            writer.p(blurb)
        if child_cls in seen:
            writer.p(f"_Recursive reference to_ `{child_cls.__name__}` _omitted._")
            continue
        render_model(writer, child_cls, full, sub_anchor, depth + 1, seen | {child_cls})

    for full, variants, field, ds in union_fields:
        sub_anchor = anchor_prefix + [full.split(".")[-1]]
        heading = f"`{full}`"
        writer.h(min(depth + 1, 6), heading, anchor=slug(sub_anchor))
        blurb = (field.description or ds or "").strip()
        if blurb:
            writer.p(blurb)
        type_field = field.discriminator or "type"
        writer.p(
            f"Discriminated union — set `{full}.{type_field}` to one of "
            + ", ".join(f"`{_type_literal(v, type_field)}`" for v in variants)
            + " and provide the matching sub-fields."
        )
        for variant in variants:
            type_literal = _type_literal(variant, type_field)
            var_anchor = sub_anchor + [type_literal or variant.__name__.lower()]
            writer.h(
                min(depth + 2, 6),
                f'`{full}.{type_field} = "{type_literal}"` ({variant.__name__})',
                anchor=slug(var_anchor),
            )
            render_model(writer, variant, full, var_anchor, depth + 2, seen | {variant})


def _type_literal(model_cls: type[BaseModel], type_field: str) -> str:
    field = model_cls.model_fields.get(type_field)
    if field is None:
        return ""
    ann = unwrap_annotated(field.annotation)
    if typing.get_origin(ann) is typing.Literal:
        return str(typing.get_args(ann)[0])
    return ""


def render_entrypoint(writer: Writer, ep: Entrypoint) -> None:
    writer.h(2, ep.title, anchor=ep.slug)
    writer.p(ep.blurb)
    writer.p(f"_Defined in_ `{ep.cls.__module__}.{ep.cls.__qualname__}`.")
    render_model(writer, ep.cls, "", [ep.slug], depth=2, seen={ep.cls})


def render_toc(writer: Writer) -> str:
    out = io.StringIO()
    out.write("## Table of Contents\n\n")
    for level, label, anchor in writer.toc:
        if level == 2:
            out.write(f"- [{label}](#{anchor})\n")
        elif level == 3:
            out.write(f"  - [{label}](#{anchor})\n")
    out.write("\n")
    return out.getvalue()


HEADER = """# Reference

This page documents every field accepted by every prime-rl entrypoint. It is
auto-generated from the Pydantic config models; do not edit by hand.

To regenerate, run from the project root:

```bash
uv run python scripts/generate_docs_reference.py
```

Each entrypoint section walks its config tree top-down. Nested sub-configs
appear under headings named after their dotted path (e.g. `trainer.model.ac`).
Discriminated unions (loss, advantage, scheduler, optimizer, …) document each
variant in turn — set the `type` field to pick one.

For conceptual context behind these knobs, see
[Configuration](configuration.md), [Training](training.md),
[Scaling](scaling.md), [Algorithms](algorithms.md), and [Advanced](advanced.md).

"""


def main() -> int:
    writer = Writer()
    writer.raw(HEADER)
    # We render entrypoints into a separate buffer first so the TOC can be
    # assembled from collected headings before being prepended.
    body = Writer()
    for ep in ENTRYPOINTS:
        render_entrypoint(body, ep)
    # Stitch: header + TOC built from body's headings + body content.
    writer.raw(render_toc(body))
    writer.raw("---\n\n")
    writer.raw(body.buf.getvalue())

    OUT_PATH.write_text(writer.buf.getvalue())
    print(f"Wrote {OUT_PATH} ({writer.buf.tell()} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
