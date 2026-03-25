from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activity_chain.backends import MockBackend, VllmBackend
from activity_chain.runner import RunConfig, run_generation


def _default_output_path() -> str:
    output_root = os.environ.get("OUTPUT_ROOT", "").strip()
    if output_root:
        return str(Path(output_root) / "activity_chain_v1.jsonl")
    return "data/activity_chain_v1.jsonl"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate activity_chain_v1 synthetic data.")
    parser.add_argument("--backend", choices=["vllm", "mock"], default=os.environ.get("ABM_BACKEND", "vllm"))
    parser.add_argument("--model", default=os.environ.get("ABM_LLM_MODEL", ""), help="Qwen model id for vLLM runs.")
    parser.add_argument("--output", default=_default_output_path())
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=_env_flag("ABM_VLLM_ENFORCE_EAGER", default=False),
        help="Disable vLLM torch.compile / cudagraph path for stability.",
    )
    parser.add_argument("--commit-every", type=int, default=100)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--mock-invalid-every",
        type=int,
        default=4,
        help="For mock backend only: every Nth record is intentionally made invalid to exercise rejects.",
    )
    parser.add_argument(
        "--archetypes",
        nargs="*",
        default=None,
        help="Limit city archetypes (e.g. dense_transit_hub sprawling_car_city). Default: all.",
    )
    args = parser.parse_args()

    if args.backend == "vllm" and not args.model:
        raise ValueError("--model is required when --backend=vllm. Set ABM_LLM_MODEL or pass --model explicitly.")
    if args.total <= 0:
        raise ValueError("--total must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.commit_every <= 0:
        raise ValueError("--commit-every must be > 0")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.mock_invalid_every < 0:
        raise ValueError("--mock-invalid-every must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    output = str(Path(args.output))
    config = RunConfig(
        output=output,
        total=args.total,
        batch_size=args.batch_size,
        seed=args.seed,
        commit_every=args.commit_every,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        archetypes=args.archetypes,
    )

    if args.backend == "mock":
        backend = MockBackend(invalid_every=args.mock_invalid_every)
    else:
        backend = VllmBackend(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
        )

    run_generation(backend=backend, config=config)


if __name__ == "__main__":
    main()
