from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activity_chain.vllm_compat import (
    get_null_subconfigs,
    patch_vllm_transformers_base_for_nullable_subconfigs,
    prepare_model_dir_for_vllm,
)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Leonardo GPU/container readiness for vLLM.")
    parser.add_argument("--model", required=True, help="Local path to the model directory on Leonardo.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--prompt", default="Reply with the single word ready.")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--text-only-multimodal", action="store_true")
    return parser.parse_args()


def _probe_triton() -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, values, mask=mask)

    n_elements = 256
    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    copy_kernel[grid](x, y, n_elements, BLOCK_SIZE=128)
    torch.cuda.synchronize()

    if not torch.equal(x, y):
        raise RuntimeError("Triton probe kernel produced incorrect output.")


def main() -> None:
    args = _parse_args()

    _print_header("Environment")
    for key in [
        "HF_HOME",
        "VLLM_CACHE_ROOT",
        "TRITON_CACHE_DIR",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "LD_LIBRARY_PATH",
        "CC",
        "CXX",
    ]:
        print(f"{key}={os.environ.get(key, '')}", flush=True)

    _print_header("GPU Visibility")
    _run(["nvidia-smi", "-L"])

    _print_header("Python Imports")
    import torch
    import transformers
    import vllm

    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"transformers={transformers.__version__}", flush=True)
    print(f"vllm={vllm.__version__}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False.")

    _print_header("libcuda")
    ctypes.CDLL("libcuda.so.1")
    print("libcuda.so.1 loaded successfully", flush=True)

    _print_header("Local Model Files")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    original_null_subconfigs = get_null_subconfigs(model_path)
    runtime_model_path = Path(prepare_model_dir_for_vllm(model_path))
    runtime_null_subconfigs = get_null_subconfigs(runtime_model_path)

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        str(runtime_model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(runtime_model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    print(f"runtime_model_path={runtime_model_path}", flush=True)
    print(f"original_null_subconfigs={original_null_subconfigs}", flush=True)
    print(f"runtime_null_subconfigs={runtime_null_subconfigs}", flush=True)
    print(f"config_sub_configs={getattr(config, 'sub_configs', None)}", flush=True)
    print(f"audio_config_type={type(getattr(config, 'audio_config', None)).__name__}", flush=True)
    print(f"vision_config_type={type(getattr(config, 'vision_config', None)).__name__}", flush=True)
    print(f"architectures={getattr(config, 'architectures', None)}", flush=True)
    print(f"tokenizer={tokenizer.__class__.__name__}", flush=True)

    _print_header("Triton JIT")
    _probe_triton()
    print("Triton JIT probe passed", flush=True)

    _print_header("vLLM Engine")
    from vllm import LLM, SamplingParams

    patch_vllm_transformers_base_for_nullable_subconfigs()

    llm_kwargs = {
        "model": str(runtime_model_path),
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    if args.text_only_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        seed=7,
    )
    outputs = llm.generate([args.prompt], sampling)
    text = outputs[0].outputs[0].text.strip()
    print(f"probe_output={text}", flush=True)

    _print_header("Result")
    print("Probe completed successfully", flush=True)


if __name__ == "__main__":
    main()
