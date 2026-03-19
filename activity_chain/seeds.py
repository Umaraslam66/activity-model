from __future__ import annotations

import random
from typing import Any, Iterator

from .schema import CITY_ARCHETYPES

ARCHETYPE_DISTRIBUTIONS: dict[str, dict[str, Any]] = {
    "dense_transit_hub": {
        "macro": {
            "transit_density_index": (0.75, 0.10),
            "active_mobility_index": (0.65, 0.12),
            "spatial_sprawl_score": (0.20, 0.10),
        },
        "parking_cost_level": {"low": 0.1, "medium": 0.3, "high": 0.6},
        "toll_level": {"none": 0.15, "moderate": 0.35, "high": 0.5},
        "dynamic_cost_sek": (80.0, 60.0),
        "network_delay_mins": (5.0, 4.0),
    },
    "sprawling_car_city": {
        "macro": {
            "transit_density_index": (0.25, 0.10),
            "active_mobility_index": (0.20, 0.10),
            "spatial_sprawl_score": (0.80, 0.10),
        },
        "parking_cost_level": {"low": 0.6, "medium": 0.3, "high": 0.1},
        "toll_level": {"none": 0.6, "moderate": 0.3, "high": 0.1},
        "dynamic_cost_sek": (30.0, 25.0),
        "network_delay_mins": (3.0, 3.0),
    },
    "winter_cycling_city": {
        "macro": {
            "transit_density_index": (0.60, 0.12),
            "active_mobility_index": (0.75, 0.10),
            "spatial_sprawl_score": (0.35, 0.12),
        },
        "parking_cost_level": {"low": 0.2, "medium": 0.5, "high": 0.3},
        "toll_level": {"none": 0.3, "moderate": 0.45, "high": 0.25},
        "dynamic_cost_sek": (50.0, 40.0),
        "network_delay_mins": (4.0, 3.5),
    },
}


def _clamp_gauss(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    return round(max(lo, min(hi, rng.gauss(mean, std))), 2)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = [weights[k] for k in keys]
    return rng.choices(keys, weights=vals, k=1)[0]


def sample_persona(rng: random.Random) -> dict[str, Any]:
    ages = [19, 27, 35, 44, 58, 72]
    income_brackets = ["low", "lower_middle", "upper_middle", "high"]
    employment_statuses = ["full_time", "part_time", "student", "retired", "unemployed"]
    household_structures = ["single", "couple", "family", "shared"]
    car_ownerships = ["none", "shared_household", "dedicated"]
    work_schedules = ["fixed_day", "shift", "flexible", "student_day", "retired", "unemployed", "remote"]

    while True:
        age = rng.choice(ages)
        employment = rng.choice(employment_statuses)
        schedule = rng.choice(work_schedules)

        if employment == "student" and schedule != "student_day":
            continue
        if employment == "retired" and schedule != "retired":
            continue
        if employment == "unemployed" and schedule != "unemployed":
            continue
        if employment in {"full_time", "part_time"} and schedule in {"student_day", "retired", "unemployed"}:
            continue
        if schedule == "remote" and employment not in {"full_time", "part_time"}:
            continue
        if age >= 67 and employment == "student":
            continue
        if age <= 22 and employment == "retired":
            continue
        break

    household = rng.choice(household_structures)
    return {
        "age": age,
        "income_bracket": rng.choice(income_brackets),
        "employment_status": employment,
        "household_structure": household,
        "has_children": household == "family" and age >= 28,
        "car_ownership": rng.choice(car_ownerships),
        "bike_ownership": rng.choice([False, True]),
        "transit_pass": rng.choice([False, True]),
        "work_schedule": schedule,
        "value_of_time_multiplier": _clamp_gauss(rng, 1.0, 0.35, 0.5, 2.0),
        "weather_resilience": _clamp_gauss(rng, 0.5, 0.25, 0.0, 1.0),
        "comfort_preference": _clamp_gauss(rng, 0.5, 0.25, 0.0, 1.0),
    }


def sample_environment(rng: random.Random, archetype: str) -> dict[str, Any]:
    dist = ARCHETYPE_DISTRIBUTIONS[archetype]

    day_types = ["weekday", "friday", "saturday", "sunday"]
    weathers = ["clear", "rain", "heavy_rain", "snow", "heatwave"]
    transit_disruptions = ["none", "minor_delay", "major_delay", "partial_shutdown"]
    congestion_levels = ["light", "moderate", "heavy", "severe"]
    special_events = ["none", "sports_event", "concert", "street_festival", "school_holiday"]

    weather = rng.choice(weathers)
    congestion = rng.choice(congestion_levels)
    if weather == "snow" and congestion == "light":
        congestion = "moderate"

    cost_mean, cost_std = dist["dynamic_cost_sek"]
    delay_mean, delay_std = dist["network_delay_mins"]

    return {
        "day_type": rng.choice(day_types),
        "weather": weather,
        "transit_disruption": rng.choice(transit_disruptions),
        "road_congestion": congestion,
        "parking_cost_level": _weighted_choice(rng, dist["parking_cost_level"]),
        "toll_level": _weighted_choice(rng, dist["toll_level"]),
        "special_event": rng.choice(special_events),
        "dynamic_cost_sek": _clamp_gauss(rng, cost_mean, cost_std, 0.0, 500.0),
        "network_delay_mins": _clamp_gauss(rng, delay_mean, delay_std, 0.0, 60.0),
    }


def sample_macro_context(rng: random.Random, archetype: str) -> dict[str, Any]:
    macro_dist = ARCHETYPE_DISTRIBUTIONS[archetype]["macro"]
    result = {}
    for field, (mean, std) in macro_dist.items():
        result[field] = _clamp_gauss(rng, mean, std, 0.0, 1.0)
    return result


def iter_scenario_seeds(
    *,
    total: int,
    seed: int,
    shard_id: int = 0,
    num_shards: int = 1,
    archetypes: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    allowed_archetypes = archetypes if archetypes else CITY_ARCHETYPES

    for idx in range(total):
        if idx % num_shards != shard_id:
            # Still consume RNG state for skipped indices to maintain determinism
            sub_rng = random.Random(rng.randint(0, 2**63))
            _ = sub_rng  # consumed
            continue

        sub_rng = random.Random(rng.randint(0, 2**63))
        archetype = sub_rng.choice(allowed_archetypes)
        persona = sample_persona(sub_rng)
        environment = sample_environment(sub_rng, archetype)
        macro_context = sample_macro_context(sub_rng, archetype)

        if persona["household_structure"] == "family" and environment["day_type"] in {"saturday", "sunday"}:
            environment["special_event"] = sub_rng.choice(["none", "school_holiday", "street_festival", "sports_event"])

        yield {
            "scenario_id": f"scenario_{idx:07d}",
            "city_archetype": archetype,
            "macro_context": macro_context,
            "persona": persona,
            "environment": environment,
        }
