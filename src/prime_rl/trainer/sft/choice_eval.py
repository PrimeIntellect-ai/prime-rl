"""Forward-only choice evals over ``simile-packed-choice-eval/v1`` exports (see ``SFTChoiceEvalConfig``).

Every rank computes the choice logits of the rows in its bins; rank 0 merges them in row-id order, writes them
to the set's step directory, hands them to the scoring function if one is configured and logs the metrics. A
``metrics.json`` in a step directory marks that step as scored.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
from torch import Tensor

from prime_rl.configs.sft import SFTChoiceEvalConfig
from prime_rl.trainer.sft.choice import supervised_rows
from prime_rl.utils.logger import get_logger

PREDICTIONS_FILE = "predictions.arrow"
MARKER_FILE = "metrics.json"

ChoiceEvalSite: TypeAlias = Literal["start", "step", "final"]
"""Where a run scores its sets: before its first optimizer update, after an optimizer step, or after the
final checkpoint."""

ChoiceEvalScoreFn: TypeAlias = Callable[..., dict[str, float]]
"""``fn(*, name, path, step, predictions, output_dir) -> metrics``, with the configured kwargs bound."""


def choice_row_ids(choice_ids: Tensor, row_ids: Tensor) -> Tensor:
    """``[M]`` row ids of the supervised positions of ``choice_ids`` ``[..., K]``, in the row order of
    ``choice_logits``; ``row_ids`` has the same leading dims."""
    return row_ids.reshape(-1)[supervised_rows(choice_ids)]


def due_choice_evals(config: SFTChoiceEvalConfig, step: int, site: ChoiceEvalSite, max_steps: int | None) -> list[str]:
    """The sets to score at ``site`` for weights with ``step`` optimizer updates.

    Step 0 is due with ``eval_on_start`` and a later step at multiples of a set's ``interval``. The final site
    scores every set, so the step site skips ``max_steps``.
    """
    if site == "final":
        return list(config.sets)
    if step == 0:
        return list(config.sets) if config.eval_on_start else []
    if site == "step" and step == max_steps:
        return []
    return [
        name
        for name, eval_set in config.sets.items()
        if eval_set.interval is not None and step % eval_set.interval == 0
    ]


def choice_eval_step_dir(run_dir: Path, name: str, step: int) -> Path:
    return run_dir / "choice_evals" / name / f"step_{step}"


def unscored_choice_evals(run_dir: Path, names: list[str], step: int) -> list[str]:
    """The sets among ``names`` whose directory for ``step`` has no ``metrics.json``."""
    return [name for name in names if not (choice_eval_step_dir(run_dir, name, step) / MARKER_FILE).exists()]


def merge_choice_predictions(parts: list[tuple[np.ndarray, np.ndarray]], num_rows: int) -> np.ndarray:
    """``[num_rows, K]`` choice logits in row-id order from per-rank ``(row_ids [m], logits [m, K])`` parts.

    Raises unless the parts hold every row id in ``0..num_rows-1`` exactly once.
    """
    row_ids = np.concatenate([part_row_ids for part_row_ids, _ in parts])
    logits = np.concatenate([part_logits for _, part_logits in parts])
    order = np.argsort(row_ids, kind="stable")
    if not np.array_equal(row_ids[order], np.arange(num_rows)):
        in_range = row_ids[(row_ids >= 0) & (row_ids < num_rows)]
        counts = np.bincount(in_range, minlength=num_rows)
        raise ValueError(
            f"Choice predictions do not cover {num_rows} rows exactly once: {int((counts == 0).sum())} missing, "
            f"{int((counts > 1).sum())} duplicated, {len(row_ids) - len(in_range)} out of range"
        )
    return logits[order]


def write_choice_predictions(path: Path, logits: np.ndarray, choice_counts: np.ndarray) -> None:
    """Write each row's ``row_id`` and fp32 ``choice_logits``, trimmed to its choice count, as an Arrow IPC file
    that atomically replaces ``path``."""
    num_rows, max_choices = logits.shape
    values = logits[np.arange(max_choices) < choice_counts[:, None]].astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(choice_counts)]).astype(np.int32)
    table = pa.table(
        {
            "row_id": pa.array(np.arange(num_rows, dtype=np.int64)),
            "choice_logits": pa.ListArray.from_arrays(pa.array(offsets), pa.array(values)),
        }
    )
    tmp_path = path.with_name(f"{path.name}.tmp")
    with pa.ipc.new_file(tmp_path, table.schema) as writer:
        writer.write_table(table)
    tmp_path.replace(path)


def score_choice_predictions(
    score_fn: ChoiceEvalScoreFn | None,
    log_fn: Callable[[dict[str, float], int], None],
    *,
    name: str,
    path: Path,
    step: int,
    output_dir: Path,
    parts: list[tuple[np.ndarray, np.ndarray]],
    choice_counts: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Merge a set's gathered predictions into ``output_dir``, score them with ``score_fn`` unless it is None,
    log the result together with ``metrics`` at ``step`` and mark the step scored.

    Any failure is logged as an error and leaves the step unmarked; it never stops training, so a broken eval
    cannot leave the other ranks waiting on a crashed rank 0.
    """
    try:
        logits = merge_choice_predictions(parts, num_rows=len(choice_counts))
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / PREDICTIONS_FILE
        write_choice_predictions(predictions, logits, choice_counts)
        scores = (
            {}
            if score_fn is None
            else score_fn(name=name, path=path, step=step, predictions=predictions, output_dir=output_dir)
        )
        row = {**scores, **metrics}
        log_fn(row, step)
        (output_dir / MARKER_FILE).write_text(json.dumps(row))
    except Exception as e:
        get_logger().error(f"Choice eval {name} at step {step} failed and stays unscored: {e!r}")
        return
    get_logger().success(f"Choice eval | {name} | Step {step} | {len(choice_counts)} rows scored")
