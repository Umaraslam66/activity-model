from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_activity_chains_qwen.py"
FIXTURE_PATH = REPO_ROOT / "spec" / "fixtures" / "activity_chain_mock_expected_first_record.json"

import math

from activity_chain.seeds import iter_scenario_seeds


def _dicts_close(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        av, bv = a[key], b[key]
        if isinstance(av, dict) and isinstance(bv, dict):
            if not _dicts_close(av, bv):
                return False
        elif isinstance(av, float) or isinstance(bv, float):
            if not math.isclose(float(av), float(bv), abs_tol=1e-6):
                return False
        elif av != bv:
            return False
    return True


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_mock_generation(*, output: Path, shard_id: int, num_shards: int, total: int) -> None:
    cmd = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--backend",
        "mock",
        "--output",
        str(output),
        "--total",
        str(total),
        "--batch-size",
        "3",
        "--commit-every",
        "1",
        "--mock-invalid-every",
        "4",
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CPU smoke tests for activity-chain generation.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "smoke")
    parser.add_argument("--total", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shard0 = output_dir / "activity_chain_mock.shard0of2.jsonl"
    shard1 = output_dir / "activity_chain_mock.shard1of2.jsonl"
    for path in [shard0, shard1, shard0.with_suffix(".rejects.jsonl"), shard1.with_suffix(".rejects.jsonl")]:
        if path.exists():
            path.unlink()

    _run_mock_generation(output=shard0, shard_id=0, num_shards=2, total=args.total)
    _run_mock_generation(output=shard1, shard_id=1, num_shards=2, total=args.total)

    shard0_records = _load_jsonl(shard0)
    shard1_records = _load_jsonl(shard1)
    shard0_rejects = _load_jsonl(shard0.with_suffix(".rejects.jsonl"))
    shard1_rejects = _load_jsonl(shard1.with_suffix(".rejects.jsonl"))

    if not shard0_records or not shard1_records:
        raise SystemExit("Smoke test failed: expected valid records in both shard outputs.")
    if not shard0_rejects and not shard1_rejects:
        raise SystemExit("Smoke test failed: expected at least one reject to exercise validator behavior.")

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if shard0_records[0] != fixture:
        raise SystemExit("Smoke test failed: first shard output does not match the deterministic mock fixture.")

    expected_shard0 = {seed["scenario_id"]: seed for seed in iter_scenario_seeds(total=args.total, seed=7, shard_id=0, num_shards=2)}
    expected_shard1 = {seed["scenario_id"]: seed for seed in iter_scenario_seeds(total=args.total, seed=7, shard_id=1, num_shards=2)}
    for record in shard0_records:
        expected = expected_shard0[record["scenario_id"]]
        if not _dicts_close(record["persona"], expected["persona"]) or not _dicts_close(record["environment"], expected["environment"]):
            raise SystemExit("Smoke test failed: shard 0 did not preserve the input seed exactly.")
        if record["city_archetype"] != expected["city_archetype"]:
            raise SystemExit("Smoke test failed: shard 0 city_archetype mismatch.")
        if not _dicts_close(record["macro_context"], expected["macro_context"]):
            raise SystemExit("Smoke test failed: shard 0 macro_context mismatch.")
    for record in shard1_records:
        expected = expected_shard1[record["scenario_id"]]
        if not _dicts_close(record["persona"], expected["persona"]) or not _dicts_close(record["environment"], expected["environment"]):
            raise SystemExit("Smoke test failed: shard 1 did not preserve the input seed exactly.")
        if record["city_archetype"] != expected["city_archetype"]:
            raise SystemExit("Smoke test failed: shard 1 city_archetype mismatch.")
        if not _dicts_close(record["macro_context"], expected["macro_context"]):
            raise SystemExit("Smoke test failed: shard 1 macro_context mismatch.")

    shard0_ids = {record["scenario_id"] for record in shard0_records}
    shard1_ids = {record["scenario_id"] for record in shard1_records}
    if shard0_ids & shard1_ids:
        raise SystemExit("Smoke test failed: shard outputs are not disjoint.")

    print(
        json.dumps(
            {
                "status": "ok",
                "shard0_valid": len(shard0_records),
                "shard1_valid": len(shard1_records),
                "shard0_rejects": len(shard0_rejects),
                "shard1_rejects": len(shard1_rejects),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
