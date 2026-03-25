from __future__ import annotations

import math
from typing import Any

import jsonschema

from .schema import ACTIVITY_CHAIN_JSON_SCHEMA, MODE_SPEED_LIMITS_KMH


def build_validator() -> jsonschema.validators.Draft202012Validator:
    return jsonschema.Draft202012Validator(ACTIVITY_CHAIN_JSON_SCHEMA)


def _speed_kmh(*, distance_km: float, travel_time_min: int) -> float:
    return distance_km / (travel_time_min / 60.0)


def _dicts_close(actual: dict, expected: dict, label: str) -> None:
    for key in expected:
        if key not in actual:
            raise ValueError(f"{label}.{key} is missing")
        a, e = actual[key], expected[key]
        if isinstance(e, float):
            if not math.isclose(a, e, abs_tol=1e-6):
                raise ValueError(f"{label}.{key} mismatch: got {a}, expected {e}")
        elif a != e:
            raise ValueError(f"{label}.{key} mismatch: got {a!r}, expected {e!r}")


def _validate_macro_context(actual: dict, expected: dict) -> None:
    _dicts_close(actual, expected, "macro_context")


def _validate_persona(actual: dict, expected: dict) -> None:
    _dicts_close(actual, expected, "persona")


def _validate_environment(actual: dict, expected: dict) -> None:
    _dicts_close(actual, expected, "environment")


def validate_activity_chain(
    record: dict[str, Any],
    *,
    validator: jsonschema.validators.Draft202012Validator,
    scenario_seed: dict[str, Any],
) -> None:
    validator.validate(record)

    if record["schema_version"] != "activity_chain_v1":
        raise ValueError("schema_version must be activity_chain_v1")
    if record["scenario_id"] != scenario_seed["scenario_id"]:
        raise ValueError("scenario_id does not match the input seed")
    if record["city_archetype"] != scenario_seed["city_archetype"]:
        raise ValueError("city_archetype does not match the input seed")
    _validate_macro_context(record["macro_context"], scenario_seed["macro_context"])
    _validate_persona(record["persona"], scenario_seed["persona"])
    _validate_environment(record["environment"], scenario_seed["environment"])

    activities = record["activities"]
    legs = record["legs"]
    persona = record["persona"]
    summary = record["summary"]

    if len(legs) != len(activities) - 1:
        raise ValueError("legs must be exactly one less than activities")
    if activities[0]["start_min"] != 0:
        raise ValueError("first activity must start at minute 0")
    if activities[-1]["end_min"] != 1440:
        raise ValueError("last activity must end at minute 1440")
    if activities[0]["purpose"] != "home" or activities[-1]["purpose"] != "home":
        raise ValueError("first and last activities must both be home")

    for idx, activity in enumerate(activities):
        if activity["sequence"] != idx:
            raise ValueError(f"activity sequence mismatch at index {idx}")
        if activity["end_min"] <= activity["start_min"]:
            raise ValueError(f"activity {idx} has non-positive duration")
        if idx > 0 and activity["start_min"] < activities[idx - 1]["end_min"]:
            raise ValueError(f"activity {idx} overlaps the previous activity")

    total_distance = 0.0
    total_travel_time = 0
    for idx, leg in enumerate(legs):
        if leg["sequence"] != idx:
            raise ValueError(f"leg sequence mismatch at index {idx}")
        if leg["from_activity_sequence"] != idx or leg["to_activity_sequence"] != idx + 1:
            raise ValueError(f"leg {idx} must connect activity {idx} to {idx + 1}")
        if len(leg["reason_tags"]) != len(set(leg["reason_tags"])):
            raise ValueError(f"leg {idx} reason_tags must be unique")

        prev_activity = activities[idx]
        next_activity = activities[idx + 1]

        if leg["depart_min"] != prev_activity["end_min"]:
            raise ValueError(f"leg {idx} depart_min must equal previous activity end_min")
        if leg["arrive_min"] != next_activity["start_min"]:
            raise ValueError(f"leg {idx} arrive_min must equal next activity start_min")

        observed_travel_time = leg["arrive_min"] - leg["depart_min"]
        if leg["travel_time_min"] != observed_travel_time:
            raise ValueError(f"leg {idx} travel_time_min mismatch")

        speed = _speed_kmh(distance_km=float(leg["distance_km"]), travel_time_min=int(leg["travel_time_min"]))
        min_speed, max_speed = MODE_SPEED_LIMITS_KMH[leg["mode"]]
        if speed < min_speed * 0.75 or speed > max_speed * 1.25:
            raise ValueError(
                f"leg {idx} mode={leg['mode']} speed={speed:.2f} km/h outside plausible range"
            )

        total_distance += float(leg["distance_km"])
        total_travel_time += int(leg["travel_time_min"])

    if persona["car_ownership"] == "none" and any(leg["mode"] == "drive" for leg in legs):
        raise ValueError("drive mode used by persona without car ownership")
    if not persona["bike_ownership"] and any(leg["mode"] == "bike" for leg in legs):
        raise ValueError("bike mode used by persona without bike ownership")

    out_of_home = sum(1 for activity in activities if activity["purpose"] != "home")
    if summary["out_of_home_activities"] != out_of_home:
        raise ValueError("summary.out_of_home_activities mismatch")

    if not math.isclose(float(summary["total_distance_km"]), total_distance, abs_tol=1.0):
        raise ValueError("summary.total_distance_km mismatch")
    if abs(int(summary["total_travel_time_min"]) - total_travel_time) > 5:
        raise ValueError("summary.total_travel_time_min mismatch")
