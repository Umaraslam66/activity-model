from __future__ import annotations

import json
from typing import Any

from .schema import SYSTEM_PROMPT


def build_user_prompt(scenario_seed: dict[str, Any]) -> str:
    return (
        "Generate one activity_chain_v1 record that exactly matches this fixed seed.\n"
        "Do not change persona, environment, city_archetype, or macro_context fields.\n"
        "Copy city_archetype and macro_context verbatim into your output.\n"
        "Use the latent persona traits (value_of_time_multiplier, weather_resilience, comfort_preference) "
        "and micro context shocks (dynamic_cost_sek, network_delay_mins) to inform mode choice and timing.\n"
        "Output schema_version=activity_chain_v1 and copy scenario_id exactly.\n"
        "Seed JSON:\n"
        f"{json.dumps(scenario_seed, ensure_ascii=False, indent=2)}\n"
    )


def build_chat_messages(scenario_seed: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(scenario_seed)},
    ]


def build_processor_messages(scenario_seed: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": build_user_prompt(scenario_seed)}]},
    ]


def build_chat_prompt(tokenizer: Any, *, scenario_seed: dict[str, Any]) -> str:
    messages = build_chat_messages(scenario_seed)
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
