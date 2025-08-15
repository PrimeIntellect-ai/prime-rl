from __future__ import annotations

import asyncio
import json
import random

# (std imports pruned)
from prime_rl.optimizer.gepa.config import GEPAConfig
from prime_rl.optimizer.gepa.evaluate import (
    PromptScore,
    score_prompt,
    score_prompt_dry_run,
    score_prompt_instances,
    score_prompt_instances_dry_run,
)
from prime_rl.optimizer.gepa.operators import crossover, mutate
from prime_rl.optimizer.gepa.reflection import reflect, reflect_llm
from prime_rl.orchestrator.config import ModelConfig as OrchestratorModelConfig
from prime_rl.utils.config import LogConfig
from prime_rl.utils.logger import format_message, format_time, get_logger, set_logger, setup_handlers
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv

# (no clean_exit usage)


async def _evaluate_population(
    population: list[str],
    cfg: GEPAConfig,
) -> list[PromptScore]:
    if cfg.dry_run:
        tasks = [score_prompt_dry_run(p, cfg.evaluate.subset_size) for p in population]
        return await asyncio.gather(*tasks)
    else:
        client_cfg = cfg.client
        model_cfg = OrchestratorModelConfig(name=cfg.model.name)
        tasks = [
            score_prompt(
                system_prompt=p,
                client_cfg=client_cfg,
                model_cfg=model_cfg,
                benchmark=cfg.evaluate.benchmark,
                subset_size=cfg.evaluate.pareto_size,
                rollouts_per_prompt=cfg.evaluate.rollouts_per_prompt,
                max_tokens=cfg.evaluate.max_tokens,
                min_tokens=cfg.evaluate.min_tokens,
            )
            for p in population
        ]
        return await asyncio.gather(*tasks)


def _default_seed_population(base_prompt: str, n: int) -> list[str]:
    variants = [
        base_prompt,
        base_prompt + "\nBe concise and avoid unnecessary verbosity.",
        base_prompt + "\nUse <final_answer> tags to present only the final answer.",
        base_prompt + "\nThink step-by-step inside <think>...</think> before answering.",
    ]
    while len(variants) < n:
        variants.append(base_prompt)
    return variants[:n]


async def gepa(cfg: GEPAConfig) -> None:
    # Setup logger
    _setup_logger(cfg.log)
    logger = get_logger()
    rng = random.Random(cfg.seed)
    monitor = setup_monitor(cfg.monitor, outputs_dir=cfg.outputs_dir, run_config=cfg)

    run_dir = cfg.outputs_dir / (cfg.run_name or "gepa")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Seed population from reverse-text system prompt as reasonable default baseline
    base_prompt = (
        "Follow the task precisely. Use <think>...</think> for internal reasoning. "
        "Output only the final answer inside <final_answer>...</final_answer>."
    )
    population = _default_seed_population(base_prompt, cfg.population_size)

    hall_of_fame: list[tuple[PromptScore, int]] = []

    rollouts_used = 0
    for gen in range(cfg.generations):
        logger.info(f"[GEPA] Generation {gen}: evaluating {len(population)} prompts")
        results = await _evaluate_population(population, cfg)

        # Persist population snapshot
        snap_path = run_dir / f"gen_{gen}.json"
        with open(snap_path, "w") as f:
            json.dump([r.__dict__ for r in results], f)

        # Log summary
        avg_scores = [r.avg_reward for r in results]
        best_idx = max(range(len(results)), key=lambda i: results[i].avg_reward)
        best = results[best_idx]
        logger.success(
            f"[GEPA] Gen {gen}: best={best.avg_reward:.3f} pass@k={best.pass_at_k} len={best.avg_completion_len:.1f}"
        )
        monitor.log({
            "gepa/gen": gen,
            "gepa/best": best.avg_reward,
            "gepa/avg": sum(avg_scores) / max(1, len(avg_scores)),
            "step": gen,
        })

        # Update Hall of Fame
        hall_of_fame.append((best, gen))
        hall_of_fame = sorted(hall_of_fame, key=lambda x: x[0].avg_reward, reverse=True)[: cfg.selection.keep_elite]

        # Early exit on final gen
        if gen == cfg.generations - 1:
            break

        # Compute Pareto scores matrix S for D_pareto
        if cfg.dry_run:
            S = [
                await score_prompt_instances_dry_run(p, cfg.evaluate.pareto_size)
                for p in population
            ]
        else:
            S = []
            for p in population:
                S.append(
                    await score_prompt_instances(
                        p,
                        cfg.client,
                        OrchestratorModelConfig(name=cfg.model.name),
                        cfg.evaluate.benchmark,
                        cfg.evaluate.pareto_size,
                        cfg.evaluate.rollouts_per_prompt,
                        cfg.evaluate.max_tokens,
                        cfg.evaluate.min_tokens,
                        offset=0,
                    )
                )

        # Pareto-front selection: choose one survivor index
        def pareto_select_index(scores: list[list[float]]) -> int:
            import math
            num_c = len(scores)
            num_i = len(scores[0]) if scores else 0
            if num_c == 0 or num_i == 0:
                return 0
            # For each instance, find max value and candidates achieving it
            max_per_i = [max(scores[c][i] for c in range(num_c)) for i in range(num_i)]
            top_sets = [
                {c for c in range(num_c) if math.isclose(scores[c][i], max_per_i[i], rel_tol=1e-9)}
                for i in range(num_i)
            ]
            union = set().union(*top_sets)
            # Non-dominated filtering within union
            def dominates(a: int, b: int) -> bool:
                ge_all = all(scores[a][i] >= scores[b][i] for i in range(num_i))
                gt_any = any(scores[a][i] > scores[b][i] for i in range(num_i))
                return ge_all and gt_any
            non_dominated = set(union)
            for a in list(union):
                for b in list(union):
                    if a != b and dominates(b, a) and a in non_dominated:
                        non_dominated.remove(a)
            # Coverage weights f: how many instances where candidate is top
            cover = {c: 0 for c in non_dominated}
            for i, tops in enumerate(top_sets):
                for c in non_dominated:
                    if c in tops:
                        cover[c] += 1
            # Weighted random choice over non_dominated by coverage
            total = sum(cover.values()) or 1
            r = rng.random() * total
            acc = 0
            for c, w in cover.items():
                acc += w
                if r <= acc:
                    return c
            return next(iter(non_dominated))

        parent_idx = pareto_select_index(S)
        survivors = [parent_idx]

        # Failures: placeholder summaries (TODO: use μ_f feedback)
        failures: list[str] = []
        if best.avg_reward < 1.0:
            failures.append("answers deviated from correct format or content")

        # Produce next generation with minibatch acceptance gate
        next_population: list[str] = []
        # Elites
        for score, _gen in hall_of_fame:
            next_population.append(score.prompt)

        # Breed survivors
        while len(next_population) < cfg.population_size and rollouts_used < cfg.budget_rollouts:
            if rng.random() < cfg.operators.crossover_rate and len(survivors) >= 2:
                a_idx, b_idx = rng.sample(survivors, 2)
                child = crossover(population[a_idx], population[b_idx], cfg.operators, rng)
            else:
                a_idx = rng.choice(survivors)
                child = population[a_idx]
            # Reflect then mutate (use LLM reflection if not dry_run)
            if cfg.dry_run:
                child = reflect(child, failures, cfg.operators, rng)
            else:
                try:
                    child = await reflect_llm(
                        cfg.client,
                        OrchestratorModelConfig(name=cfg.model.name),
                        child,
                        list(failures),
                        cfg.operators.max_prompt_chars,
                    )
                except Exception:
                    child = reflect(child, failures, cfg.operators, rng)
            if rng.random() < cfg.operators.mutation_rate:
                child = mutate(child, cfg.operators, rng)
            # Acceptance: compare on a minibatch from feedback pool
            if cfg.dry_run:
                parent_scores = await score_prompt_instances_dry_run(population[a_idx], cfg.minibatch_size)
                child_scores = await score_prompt_instances_dry_run(child, cfg.minibatch_size)
            else:
                parent_scores = await score_prompt_instances(
                    population[a_idx], cfg.client, OrchestratorModelConfig(name=cfg.model.name),
                    cfg.evaluate.benchmark, cfg.minibatch_size, cfg.evaluate.rollouts_per_prompt,
                    cfg.evaluate.max_tokens, cfg.evaluate.min_tokens, offset=cfg.evaluate.pareto_size,
                )
                child_scores = await score_prompt_instances(
                    child, cfg.client, OrchestratorModelConfig(name=cfg.model.name),
                    cfg.evaluate.benchmark, cfg.minibatch_size, cfg.evaluate.rollouts_per_prompt,
                    cfg.evaluate.max_tokens, cfg.evaluate.min_tokens, offset=cfg.evaluate.pareto_size,
                )

            rollouts_used += cfg.minibatch_size
            if sum(child_scores) / len(child_scores) > sum(parent_scores) / len(parent_scores):
                next_population.append(child)

        population = next_population[: cfg.population_size]

        if rollouts_used >= cfg.budget_rollouts:
            logger.info("[GEPA] Budget exhausted; stopping evolution")
            break

    # Export best prompt
    best_overall = hall_of_fame[0][0] if hall_of_fame else results[0]
    out_path = run_dir / "best_prompt.txt"
    with open(out_path, "w") as f:
        f.write(best_overall.prompt)
    logger.success(f"[GEPA] Exported best prompt to {out_path}")


def main():
    asyncio.run(gepa(parse_argv(GEPAConfig)))


if __name__ == "__main__":
    main()


def _setup_logger(log_config: LogConfig):
    if get_logger():
        return
    fmt = format_time(log_config) + format_message()
    logger = setup_handlers(__import__("loguru").logger, fmt, log_config, rank=0)
    set_logger(logger)


