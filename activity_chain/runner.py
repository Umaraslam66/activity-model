from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .backends import GenerationBackend
from .seeds import iter_scenario_seeds
from .validation import build_validator, validate_activity_chain


@dataclass(frozen=True)
class RunConfig:
    output: str
    total: int
    batch_size: int
    seed: int
    commit_every: int
    shard_id: int
    num_shards: int
    archetypes: list[str] | None = None


def run_generation(*, backend: GenerationBackend, config: RunConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [activity-chain] %(message)s")

    out_path = Path(config.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = out_path.with_suffix(".rejects.jsonl")

    validator = build_validator()
    completed = 0
    rejected = 0
    started = time.perf_counter()
    scenario_seed_iter = iter_scenario_seeds(
        total=config.total,
        seed=config.seed,
        shard_id=config.shard_id,
        num_shards=config.num_shards,
        archetypes=config.archetypes,
    )

    with (
        out_path.open("a", encoding="utf-8", newline="\n") as f_out,
        rejects_path.open("a", encoding="utf-8", newline="\n") as f_rejects,
    ):
        while True:
            batch = []
            for _ in range(config.batch_size):
                try:
                    batch.append(next(scenario_seed_iter))
                except StopIteration:
                    break
            if not batch:
                break

            responses = backend.generate_batch(batch)
            for scenario_seed, response in zip(batch, responses, strict=True):
                try:
                    record = json.loads(response.raw_text)
                    validate_activity_chain(record, validator=validator, scenario_seed=scenario_seed)
                except Exception as exc:  # noqa: BLE001
                    rejected += 1
                    f_rejects.write(
                        json.dumps(
                            {
                                "scenario_seed": scenario_seed,
                                "error": str(exc),
                                "raw": response.raw_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1

            if (completed + rejected) % config.commit_every == 0:
                f_out.flush()
                f_rejects.flush()

            elapsed = time.perf_counter() - started
            logging.info(
                "progress shard=%s/%s completed=%s rejected=%s seen=%s eps=%.3f",
                config.shard_id,
                config.num_shards,
                completed,
                rejected,
                completed + rejected,
                completed / elapsed if elapsed > 0 else 0.0,
            )

    elapsed = time.perf_counter() - started
    logging.info(
        "done shard=%s/%s completed=%s rejected=%s elapsed_s=%.1f eps=%.3f output=%s rejects=%s",
        config.shard_id,
        config.num_shards,
        completed,
        rejected,
        elapsed,
        completed / elapsed if elapsed > 0 else 0.0,
        out_path,
        rejects_path,
    )
