from __future__ import annotations

import json
from pathlib import Path


def patch_vllm_transformers_base_for_nullable_subconfigs() -> None:
    """Patch vLLM's transformers backend to tolerate nullable sub-configs.

    Gemma 4 config can include entries such as `audio_config: null` while still
    advertising that field via `sub_configs`. Some vLLM builds unconditionally
    dereference each sub-config's `dtype`, which crashes before model load.
    """

    import torch
    from vllm.model_executor.models.transformers.base import Base

    if getattr(Base, "_bonzai_nullable_subconfigs_patch", False):
        return

    def _safe_patch_config(self) -> None:
        self.text_config._attn_implementation = "vllm"
        self.config.dtype = torch.get_default_dtype()
        for sub_config_name in getattr(self.config, "sub_configs", {}):
            sub_config = getattr(self.config, sub_config_name, None)
            if sub_config is None:
                continue
            if getattr(sub_config, "dtype", None) != (dtype := self.config.dtype):
                sub_config.dtype = dtype

    Base._patch_config = _safe_patch_config
    Base._bonzai_nullable_subconfigs_patch = True


def get_null_subconfigs(model: str | Path) -> list[str]:
    model_path = Path(model).resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return []

    config_data = json.loads(config_path.read_text())
    sub_configs = config_data.get("sub_configs")
    if not isinstance(sub_configs, dict):
        return []

    return [name for name in sub_configs if config_data.get(name) is None]


def prepare_model_dir_for_vllm(model: str | Path) -> str:
    """Create a sanitized model directory for vLLM if config contains null sub-configs.

    vLLM worker processes load config directly from disk. Preparing a derived model
    directory avoids relying on in-process monkey patches that do not propagate to
    spawned workers.
    """

    model_path = Path(model).resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return str(model_path)

    config_data = json.loads(config_path.read_text())
    sub_configs = config_data.get("sub_configs")
    if not isinstance(sub_configs, dict):
        return str(model_path)

    null_subconfigs = get_null_subconfigs(model_path)
    if not null_subconfigs:
        return str(model_path)

    sanitized_path = model_path.parent / f"{model_path.name}--bonzai-vllm"
    sanitized_path.mkdir(parents=True, exist_ok=True)

    for child in model_path.iterdir():
        target = sanitized_path / child.name
        if child.name == "config.json":
            continue
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(child, target_is_directory=child.is_dir())

    sanitized_config = dict(config_data)
    sanitized_config["sub_configs"] = dict(sub_configs)
    for name in null_subconfigs:
        sanitized_config[name] = {}

    (sanitized_path / "config.json").write_text(json.dumps(sanitized_config, indent=2) + "\n")
    return str(sanitized_path)
