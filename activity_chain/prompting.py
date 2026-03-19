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


def build_chat_prompt(tokenizer: Any, *, scenario_seed: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(scenario_seed)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
