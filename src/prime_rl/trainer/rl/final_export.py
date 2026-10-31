"""GeneBench round-2 end-of-run export inside the RL trainer (GB_FINAL_EXPORT; flow and files in
final_export_plan.py).

After the final checkpoint save, every rank enters ``FinalExport.run``:

1. The master waits for the final eval and for every eval still pending for a kept checkpoint (file polling on
   the orchestrator's eval records, up to GB_FINAL_EVAL_TIMEOUT_S), pruning with keep-3 as evals land. Then it
   settles the plan (role holders -> repos), prunes to the final keep set and writes final_checkpoints.json.
   The other ranks wait on the c10d store meanwhile, not in an NCCL collective, whose watchdog would kill them
   after ``dist_timeout_seconds`` (evals take hours).
2. Every rank exports each planned local step with trainer/sft/hf_export.py: the final step from the live
   weights first, then the others loaded weights-only from their DCP checkpoints into the same model. Tensors are
   rounded to the base dtypes on the GPU before they reach host memory. Each repo is private, refused if it holds
   another run's or step's export, and verified by sha256; then its staged shards are deleted.
3. The master prunes once more (the final step is the resume point once the orchestrator's teardown checkpoint is
   written), then writes hf-export/DONE, or hf-export/FAILED with the errors. A failure is loud but never raises:
   a non-zero trainer exit can restart the whole run, while the checkpoints are safe on the volume either way.

A restarted trainer that resumes at ``max_steps`` skips training and only finishes the export: the plan file
keeps the steps and roles, and state.json skips verified uploads. An export that crashes the trainer (a rank
stuck in a collective until the NCCL watchdog fires) is tried at most MAX_EXPORT_ATTEMPTS times; then FAILED.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from prime_rl.trainer.rl import final_export_plan as fx
from prime_rl.trainer.sft import hf_export
from prime_rl.trainer.sft import rl_export_plan as plan
from prime_rl.trainer.world import get_world, wait_for_master
from prime_rl.utils.ckpt_retention import (
    CKPT_COMPLETE_MARKERS,
    EVAL_RESULTS_DIR,
    WRITER_FILE,
    Keep3Retention,
    atomic_write_json,
    read_records,
)
from prime_rl.utils.gb_settings import EXPORT_REPO_PREFIX, FINAL_EXPORT, GBSettings, GBSettingsError
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.trainer import TrainerConfig

# How long the other ranks wait for the master's plan beyond the eval wait: the final prune removes up to a few
# 1.45 TB checkpoint directories, at a speed nobody has measured on the hosted volume.
PLAN_MARGIN_S = 4 * 3600
POLL_S = 30.0
LOG_EVERY_S = 600.0
PLAN_KEY = "gb_final_export/plan"
# After the evals, how long the master waits for the orchestrator's final checkpoint (written at its teardown,
# seconds after the last eval) before the final prune. Until it exists, the newest step with both markers is
# the resume point and is kept.
MARKER_WAIT_S = 1800.0
MAX_EXPORT_ATTEMPTS = 2


def _host_rss_gib() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return None


def release_optimizer_state(optimizer, logger=None) -> None:
    """Training is over once the final checkpoint is saved: drop the optimizer state (AdamW moments, offloaded to
    pinned host memory with offload="optimizer", ~240 GB per node here) so the export's host buffers have room.
    CPUOffloadOptimizer also holds the pinned buffers in ``_cpu_buffers``; they are only freed once both go."""
    rss_before = _host_rss_gib()
    base = getattr(optimizer, "base_optimizer", optimizer)  # an OffloadOptimizer wraps the torch optimizer
    base.state.clear()
    cpu_buffers = getattr(optimizer, "_cpu_buffers", None)
    if isinstance(cpu_buffers, dict):
        cpu_buffers.clear()
    gc.collect()
    torch.cuda.empty_cache()
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    rss_after = _host_rss_gib()
    if logger is not None and rss_before is not None and rss_after is not None:
        logger.info(f"Final export: released the optimizer state (host RSS {rss_before:.1f} -> {rss_after:.1f} GiB)")


class FinalExport:
    """Created on every rank when GB_FINAL_EXPORT is on. ``startup`` runs before the model is built (fail fast);
    ``run`` after the final checkpoint save. ``retention`` is the master's Keep3Retention (None elsewhere)."""

    def __init__(
        self,
        config: TrainerConfig,
        gb: GBSettings,
        *,
        ckpt_dir: Path,
        retention: Keep3Retention | None,
        external_steps: set[int],
        fork_dir: Path | None,
    ) -> None:
        self.config = config
        self.gb = gb
        self.ckpt_dir = ckpt_dir
        self.retention = retention
        self.external_steps = set(external_steps)
        # The forked step's directory (resume.dir), also on a restart that resumed from this run's own checkpoint
        self.fork_dir = fork_dir or getattr(getattr(config, "resume", None), "dir", None)
        self.paths = fx.ExportPaths(config.output_dir)
        self.run_id = config.output_dir.name
        self.base_dir = Path(config.model.name)
        self.max_steps = config.max_steps
        # With offload="full" the GPU holds mixed-precision compute copies and the fp32 masters live on the CPU,
        # so the final step is exported from its checkpoint like the others.
        self.live_ok = config.model.offload != "full"
        self.live_step: int | None = None
        self.logger = get_logger()
        self.world = get_world()

    # ── startup ────────────────────────────────────────────────────────────

    def startup(self, resume_step: int | None, fork_dir: Path | None) -> bool:
        """Collective. Checks what would otherwise fail only after training, clears an earlier timeline's export
        files when this attempt will train, and arms the orchestrator's wait. Returns True when this attempt
        resumed from the final checkpoint, so training is done and only the export is left."""
        if self.max_steps is None:
            raise GBSettingsError(f"{FINAL_EXPORT}=1 exports at the end of training and needs max_steps")
        if self.external_steps and max(self.external_steps) >= self.max_steps:
            raise GBSettingsError(
                f"max_steps={self.max_steps} is not after the forked step {max(self.external_steps)} (max_steps is a total)"
            )
        if not os.environ.get("HF_TOKEN") and self.config.env_vars.get("HF_TOKEN"):
            # huggingface_hub reads the token from the environment only
            os.environ["HF_TOKEN"] = self.config.env_vars["HF_TOKEN"]
        export_only = resume_step is not None and fork_dir is None and resume_step >= self.max_steps
        error = [None]
        if self.world.is_master:
            try:
                self._master_startup(resume_step, export_only)
            except Exception as e:
                error[0] = f"{type(e).__name__}: {e}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise GBSettingsError(f"final export startup check failed: {error[0]}")
        if export_only:
            self.logger.info(
                f"Resumed from the final checkpoint (step {resume_step} >= max_steps={self.max_steps}): skipping "
                "training, finishing the final export"
            )
        return export_only

    def _master_startup(self, resume_step: int | None, export_only: bool) -> None:
        hf_export.check_base(self.base_dir)
        self._check_token()
        if not export_only:
            for line in fx.invalidate_stale(self.paths, resume_step, self.max_steps, self.logger):
                self.logger.info(f"Final export: {line}")
        else:
            # An earlier attempt's outcome would release the orchestrator while this attempt exports again. The
            # orchestrator reads it only after the startup broadcast, which comes after this.
            for line in fx.clear_outcome(self.paths):
                self.logger.info(f"Final export: {line}")
        fx.write_armed(
            self.paths,
            {"final_step": self.max_steps, "run_id": self.run_id, "repo_prefix": self.gb.export_repo_prefix},
        )
        self.logger.info(
            f"Final export armed: after step {self.max_steps} and its eval, the keep-3 role holders go to private "
            f"repos {self.gb.export_repo_prefix}-step<N> (base model {self.base_dir})"
        )

    def _check_token(self) -> None:
        """The token works and can write under the repo prefix's namespace. A Hub outage only warns: the export
        retries at the end anyway."""
        from huggingface_hub import HfApi

        namespace = self.gb.export_repo_prefix.split("/", 1)[0]
        try:
            who = hf_export.hub("whoami", HfApi().whoami)
        except Exception as e:
            if plan.is_transient(e):
                self.logger.warning(f"Final export: Hugging Face whoami failed ({type(e).__name__}: {e}); continuing")
                return
            raise
        problem, warning = fx.token_write_problem(who, namespace)
        if problem is not None:
            raise GBSettingsError(f"{problem} ({EXPORT_REPO_PREFIX}={self.gb.export_repo_prefix!r})")
        if warning is not None:
            self.logger.warning(f"Final export: {warning}")
        self.logger.info(f"Final export: Hugging Face token of {who.get('name')!r} can write to {namespace!r}")

    # ── end of training ────────────────────────────────────────────────────

    def run(self, model: torch.nn.Module, optimizer, final_step: int, heart=None) -> bool:
        """Collective; see the module docstring. Returns True when every planned step is verified in its repo."""
        group = dist.new_group(backend="gloo", timeout=hf_export.GLOO_TIMEOUT)
        if self.live_ok:
            self.live_step = final_step
        if self.config.model.offload != "full":
            release_optimizer_state(optimizer, self.logger)

        box: list = [None]
        if self.world.is_master:
            try:
                box[0] = self._settle(final_step, heart)
            except Exception as e:
                self.logger.opt(exception=True).error("Final export: settling the plan failed")
                box[0] = {"error": f"{type(e).__name__}: {e}", "final_step": final_step}
        wait_for_master(PLAN_KEY, self.gb.final_eval_timeout_s + PLAN_MARGIN_S)
        dist.broadcast_object_list(box, src=0, group=group)
        export_plan = box[0]
        if "error" in export_plan:
            if self.world.is_master:
                path = fx.write_outcome(
                    self.paths, False, {"final_step": final_step, "errors": {"plan": export_plan["error"]}}
                )
                self.logger.error(f"FINAL EXPORT FAILED before exporting anything: {export_plan['error']} ({path})")
            dist.barrier(group=group)
            return False

        state_box: list = [None]
        if self.world.is_master:
            state = fx.load_state(self.paths.state)
            gave_up = fx.begin_export_attempt(state, fx.export_actions(export_plan, state), MAX_EXPORT_ATTEMPTS)
            try:
                atomic_write_json(self.paths.state, state)
            except OSError as e:
                self.logger.error(f"Final export: failed to record the export attempt: {e}")
            state_box[0] = (state, gave_up)
        dist.broadcast_object_list(state_box, src=0, group=group)
        state, gave_up = state_box[0]
        if gave_up is not None:
            if self.world.is_master:
                path = fx.write_outcome(self.paths, False, {"final_step": final_step, "errors": {"attempts": gave_up}})
                self.logger.error(f"FINAL EXPORT FAILED: {gave_up} ({path})")
            dist.barrier(group=group)
            return False
        actions = fx.export_actions(export_plan, state)
        self.logger.info(f"Final export actions: {[(item['step'], item['role'], action) for item, action in actions]}")

        ctx, ctx_error = None, None
        if any(action == fx.EXPORT for _, action in actions):
            try:  # raises on every rank together (run_guarded)
                ctx = hf_export.prepare_context(group, self.base_dir, cast_on_gpu=True)
            except Exception as e:
                ctx_error = f"preparing the base model failed: {type(e).__name__}: {e}"
        report = self._load_report(export_plan) if self.world.is_master else None
        card = self._card_facts()

        def export(item: dict) -> str:
            if ctx_error is not None:
                raise RuntimeError(ctx_error)
            spec = self._spec(item)
            load_dcp = not (item["source"] == fx.LIVE and self.live_step == item["step"])
            if load_dcp:
                self.live_step = None  # the live weights are replaced from here on
            commit, entry = hf_export.export_one(
                model,
                spec,
                ctx,
                self.paths.root,
                group,
                load_dcp=load_dcp,
                heldout=item["heldout"],
                card_kwargs=self._card_kwargs(item, card),
                entry_extra=self._entry_extra(item, final_step, load_dcp),
                preflight=True,
                delete_shards=True,
            )
            if load_dcp:
                self.live_step = item["step"]
            if self.world.is_master:
                # Master-only bookkeeping must not raise: the ranks would leave the loop's collectives out of step
                try:
                    report["exports"] = [e for e in report["exports"] if e.get("step") != item["step"]] + [entry]
                    plan.write_json(self.paths.report, report)
                    self.logger.success(plan.done_line(entry))
                except Exception as e:
                    self.logger.error(f"Final export: failed to update {self.paths.report}: {type(e).__name__}: {e}")
            return commit

        def refresh(item: dict) -> str:
            return self._refresh_card(item, card, group)

        def on_verified(item: dict, commit: str) -> None:
            if self.world.is_master:
                fx.record_upload(state, item, commit)
                try:
                    atomic_write_json(self.paths.state, state)
                    atomic_write_json(self.paths.final_checkpoints, fx.final_checkpoints_doc(export_plan, state))
                except OSError as e:
                    self.logger.error(f"Final export: failed to record the upload of step {item['step']}: {e}")

        errors = fx.run_actions(actions, export=export, refresh=refresh, on_verified=on_verified, log=self.logger)
        ok = not errors and not export_plan.get("problems")
        if self.world.is_master:
            try:
                self._finish(export_plan, state, errors, ok, report)
            except Exception:
                self.logger.opt(exception=True).error("Final export: writing the outcome failed")
        dist.barrier(group=group)
        return ok

    def _settle(self, final_step: int, heart) -> dict:
        """Master: wait for the evals, then fix the plan, prune to the final keep set, and write
        final_checkpoints.json."""
        assert self.retention is not None
        self._warn_if_orchestrator_not_waiting()
        retention, logger = self.retention, self.logger
        # An earlier attempt's plan is kept as is (same steps, same roles), so its steps must survive the prunes
        # during the wait too, even if the records moved since it was written.
        existing = fx.read_json(self.paths.plan)
        if fx.plan_matches(existing, final_step):
            retention.protected_steps |= fx.plan_steps(existing)
        last_log = [-LOG_EVERY_S]
        pending: list[int] = []

        def evals_done() -> bool:
            nonlocal pending
            result = retention.run_safely(final_step, assume_complete=[final_step])
            sel = result.selection if result is not None else retention.select([final_step], final_step)[1]
            pending = sel.pending
            return not pending

        def on_wait(waited: float) -> None:
            if heart is not None:
                heart.beat()
            if waited - last_log[0] >= LOG_EVERY_S:
                last_log[0] = waited
                logger.info(
                    f"Final export: waiting for the evals of step(s) {pending} ({waited / 3600:.1f} h of "
                    f"{self.gb.final_eval_timeout_s / 3600:.1f} h)"
                )

        logger.info(f"Final export: waiting for the final eval (step {final_step}) and any pending eval of a kept step")
        finished = fx.wait_for(evals_done, timeout_s=self.gb.final_eval_timeout_s, poll_s=POLL_S, on_wait=on_wait)
        if not finished:
            logger.error(
                f"Final export: evals of step(s) {pending} did not finish within {self.gb.final_eval_timeout_s:.0f}s. "
                "They are kept on the volume but cannot hold a best role; exporting the current role holders"
            )
        else:
            # The final prune keeps the older resume point until the final step has both markers (see
            # Keep3Retention.select); the orchestrator writes its marker seconds after the last eval.
            marker = self.ckpt_dir / f"step_{final_step}" / "orchestrator" / CKPT_COMPLETE_MARKERS["orchestrator"]

            def beat(_: float) -> None:
                if heart is not None:
                    heart.beat()

            if not fx.wait_for(marker.exists, timeout_s=MARKER_WAIT_S, poll_s=POLL_S, on_wait=beat):
                logger.warning(
                    f"Final export: the orchestrator's checkpoint of step {final_step} ({marker}) is not written "
                    f"after {MARKER_WAIT_S:.0f}s; the previous complete step stays as the resume point"
                )

        if fx.plan_matches(existing, final_step):
            export_plan = existing
            logger.info(f"Final export: reusing the plan of an earlier attempt ({self.paths.plan})")
        else:
            _, sel = retention.select([final_step], final_step)
            records = read_records(self.ckpt_dir, self.gb.eval_env, logger)
            export_plan = fx.build_plan(
                sel,
                final_step=final_step,
                external_steps=self.external_steps,
                records=records,
                eval_env=self.gb.eval_env,
                prefix=self.gb.export_repo_prefix,
                run_id=self.run_id,
                ckpt_dir=self.ckpt_dir,
                timed_out=not finished,
                pending=sel.pending,
            )
            for item in export_plan["external"]:
                item["repo_check"] = self._check_external_repo(item["repo"])
            atomic_write_json(self.paths.plan, export_plan)
        if export_plan.get("problems"):
            logger.error(f"Final export: plan problems: {export_plan['problems']}")

        # The planned steps survive the final prune even if the records changed since the plan was written.
        retention.protected_steps |= fx.plan_steps(export_plan)
        retention.run_safely(final_step, assume_complete=[final_step])
        state = fx.load_state(self.paths.state)
        atomic_write_json(self.paths.final_checkpoints, fx.final_checkpoints_doc(export_plan, state))
        logger.info(
            "Final export plan: "
            + ", ".join(f"step {i['step']} ({i['role']}) -> {i['repo']}" for i in export_plan["exports"])
            + "".join(f"; step {i['step']} ({i['role']}) is external: {i['repo']}" for i in export_plan["external"])
        )
        return export_plan

    def _warn_if_orchestrator_not_waiting(self) -> None:
        writer = fx.read_json(self.ckpt_dir / EVAL_RESULTS_DIR / WRITER_FILE) or {}
        if not (writer.get("settings") or {}).get("final_export"):
            self.logger.warning(
                f"Final export: the orchestrator did not report {FINAL_EXPORT}=1 ({EVAL_RESULTS_DIR}/{WRITER_FILE}), "
                "so it will not wait for the export before it finalizes the run; the platform may tear the run down "
                "mid-upload. Set it in [orchestrator.env_vars] too"
            )

    def _check_external_repo(self, repo_id: str) -> dict:
        """A forked step's repo is never uploaded again; record that it exists (best effort)."""
        from huggingface_hub import HfApi

        try:
            info = hf_export.hub(f"repo_info {repo_id}", HfApi().repo_info, repo_id)
            return {"exists": True, "sha": info.sha, "private": info.private}
        except Exception as e:
            self.logger.warning(f"Final export: could not check the external repo {repo_id}: {type(e).__name__}: {e}")
            return {"exists": None, "error": f"{type(e).__name__}: {e}"}

    def _spec(self, item: dict) -> plan.ExportSpec:
        return plan.ExportSpec(self.run_id, item["step"], item["repo"], item["role"], dcp_path=Path(item["dcp_dir"]))

    def _card_facts(self) -> dict:
        """Facts for the model card that only the orchestrator knows (its writer.json)."""
        writer = fx.read_json(self.ckpt_dir / EVAL_RESULTS_DIR / WRITER_FILE) or {}
        return {
            "batch_size": writer.get("batch_size"),
            "eval_group_size": (writer.get("eval_group_size") or {}).get(self.gb.eval_env),
        }

    def _card_kwargs(self, item: dict, card: dict) -> dict:
        source = f"hosted RL run `{self.run_id}`, step {item['step']}"
        if self.external_steps and self.fork_dir is not None:
            source += f" (continued from step {max(self.external_steps)} of run `{self.fork_dir.parents[1].name}`)"
        elif self.external_steps:
            source += f" (continued from step {max(self.external_steps)} of an earlier run)"
        rollouts = f"{card['batch_size']} rollouts per step" if card.get("batch_size") else "rollouts per step n/a"
        eval_desc, pass_label = fx.card_eval_desc(item.get("eval"), card.get("eval_group_size"))
        return {
            "source": source,
            "eval_desc": eval_desc,
            "pass_at_label": pass_label,
            "training": (
                f"GRPO, {rollouts} (16 per task), AdamW at learning rate {self.config.optim.lr:g}, RLM agent harness\n"
                "  (bash, edit, ipython), held-out split of 20 of the 121 tasks"
            ),
        }

    def _entry_extra(self, item: dict, final_step: int, load_dcp: bool) -> dict:
        rec = item.get("eval") or {}
        return {
            "roles": item["roles"],
            "final_step": final_step,
            "weights_from": "dcp checkpoint" if load_dcp else "live trainer model (the final step)",
            "rounding": "to the base dtype on the GPU, so relative_change_fp32 equals relative_change",
            "heldout_eval_record": {k: rec.get(k) for k in ("status", "eval_error", "n_episodes", "policy_version")},
        }

    def _load_report(self, export_plan: dict) -> dict:
        report = fx.read_json(self.paths.report) or {}
        if report.get("final_step") != export_plan["final_step"]:
            report = {}
        report.setdefault("run_id", self.run_id)
        report.setdefault("final_step", export_plan["final_step"])
        report.setdefault(
            "base", {"dir": str(self.base_dir), "repo": hf_export.BASE_REPO, "revision": hf_export.BASE_REVISION}
        )
        report.setdefault("exports", [])
        report["plan"] = {k: export_plan.get(k) for k in ("roles", "eval_wait_timed_out", "pending_evals", "problems")}
        report["all_done"] = False
        return report

    def _refresh_card(self, item: dict, card: dict, group) -> str:
        """Collective. A step verified in its repo under another role (an earlier timeline's plan): rewrite and
        commit only its README and export report. Returns the new head commit."""
        spec = self._spec(item)

        def commit_card() -> str | None:
            if not self.world.is_master:
                return None
            from huggingface_hub import CommitOperationAdd, HfApi

            out_dir = self.paths.staging(item["step"])
            detail = json.loads((out_dir / plan.REPORT_FILE).read_text())
            detail.update(role=item["role"], roles=item["roles"])
            (out_dir / "README.md").write_text(
                plan.model_card(
                    spec, item["heldout"], hf_export.BASE_REPO, hf_export.BASE_REVISION, **self._card_kwargs(item, card)
                )
            )
            plan.write_json(out_dir / plan.REPORT_FILE, detail)
            info = hf_export.hub(
                f"role commit to {spec.repo_id}",
                HfApi().create_commit,
                spec.repo_id,
                operations=[CommitOperationAdd(name, out_dir / name) for name in ("README.md", plan.REPORT_FILE)],
                commit_message=f"Role: {item['role']}",
            )
            return info.oid

        box = [hf_export.run_guarded(f"[{spec.name}] role refresh of {spec.repo_id}", group, commit_card)]
        dist.broadcast_object_list(box, src=0, group=group)
        return box[0]

    def _finish(self, export_plan: dict, state: dict, errors: dict[int, str], ok: bool, report: dict) -> None:
        """Master: a last prune, final_checkpoints.json, the run-level report, and the DONE or FAILED marker."""
        # Before the outcome, which lets the orchestrator finalize: the final step's orchestrator marker is written
        # by now (it comes before the orchestrator waits for the outcome), so the old resume point can go.
        if self.retention is not None:
            self.retention.run_safely(export_plan["final_step"], assume_complete=[export_plan["final_step"]])
            export_plan = dict(
                export_plan,
                kept_not_exported={
                    s: r
                    for s, r in (export_plan.get("kept_not_exported") or {}).items()
                    if (self.ckpt_dir / f"step_{s}").is_dir()
                },
            )
        atomic_write_json(self.paths.final_checkpoints, fx.final_checkpoints_doc(export_plan, state, errors))
        report["all_done"] = ok
        plan.write_json(self.paths.report, report)
        payload = fx.outcome_payload(export_plan, state, errors)
        path = fx.write_outcome(self.paths, ok, payload)
        if ok:
            self.logger.success(f"Final export done: {json.dumps(payload['repos'], sort_keys=True)} ({path})")
            print(f"{plan.ALL_DONE_MARKER} {json.dumps(payload, sort_keys=True, separators=(',', ':'))}", flush=True)
        else:
            self.logger.error(
                f"FINAL EXPORT FAILED: errors {payload['errors']}, problems {payload['problems']} ({path}). The "
                "checkpoints stay on the volume; export them with the standalone export run"
            )
