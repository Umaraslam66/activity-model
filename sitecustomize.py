from __future__ import annotations

# Ensure the vLLM nullable-subconfig compatibility patch is applied as soon as
# Python starts, including in worker processes spawned by vLLM.
try:
    from activity_chain.vllm_compat import patch_vllm_transformers_base_for_nullable_subconfigs

    patch_vllm_transformers_base_for_nullable_subconfigs()
except Exception:
    # Avoid breaking unrelated Python entrypoints if vLLM is not installed or
    # if the compatibility patch is not relevant in the current environment.
    pass
