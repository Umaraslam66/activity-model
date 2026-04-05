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


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _is_local_model_path(model: str) -> bool:
    return Path(model).expanduser().exists()


def _resolve_torch_dtype(dtype: str):
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _model_input_device(model) -> object:
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Could not determine a model input device.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Leonardo GPU/container readiness for Gemma 4 via Transformers.")
    parser.add_argument("--model", required=True, help="Local path to the model directory on Leonardo.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt", default="Reply with the single word ready.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    _print_header("Environment")
    for key in [
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    ]:
        print(f"{key}={os.environ.get(key, '')}", flush=True)

    _print_header("GPU Visibility")
    _run(["nvidia-smi", "-L"])

    _print_header("Python Imports")
    import accelerate
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"transformers={transformers.__version__}", flush=True)
    print(f"accelerate={accelerate.__version__}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False.")

    _print_header("libcuda")
    ctypes.CDLL("libcuda.so.1")
    print("libcuda.so.1 loaded successfully", flush=True)

    _print_header("Local Model Files")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    local_files_only = _is_local_model_path(args.model)
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    print(f"model_path={model_path}", flush=True)
    print(f"config_class={config.__class__.__name__}", flush=True)
    print(f"model_type={getattr(config, 'model_type', None)}", flush=True)
    print(f"architectures={getattr(config, 'architectures', None)}", flush=True)
    print(f"processor={processor.__class__.__name__}", flush=True)
    print(f"tokenizer={tokenizer.__class__.__name__}", flush=True)

    _print_header("Transformers Model Load")
    model_kwargs = {
        "torch_dtype": _resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        **model_kwargs,
    )
    model.eval()
    input_device = _model_input_device(model)
    print(f"input_device={input_device}", flush=True)
    print(f"hf_device_map={getattr(model, 'hf_device_map', None)}", flush=True)

    _print_header("Generation Probe")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
    ]
    try:
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_model_len,
    )
    inputs = inputs.to(input_device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    print(f"probe_output={text}", flush=True)

    _print_header("Result")
    print("Probe completed successfully", flush=True)


if __name__ == "__main__":
    main()
