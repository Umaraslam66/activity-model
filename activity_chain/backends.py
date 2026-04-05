from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .prompting import build_chat_prompt, build_processor_messages
from .schema import ACTIVITY_CHAIN_JSON_SCHEMA, REASON_TAGS, ZONE_CHOICES
from .vllm_compat import patch_vllm_transformers_base_for_nullable_subconfigs, prepare_model_dir_for_vllm


@dataclass(frozen=True)
class BackendResponse:
    raw_text: str


class GenerationBackend(Protocol):
    def generate_batch(self, scenario_seeds: list[dict[str, Any]]) -> list[BackendResponse]:
        ...


def _is_local_model_path(model: str) -> bool:
    return Path(model).expanduser().exists()


def _resolve_torch_dtype(dtype: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for transformers backend: {dtype}")
    return mapping[dtype]


def _model_input_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Could not determine a model input device.")


class VllmBackend:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_model_len: int,
        tensor_parallel_size: int,
        dtype: str,
        seed: int,
        enforce_eager: bool,
        trust_remote_code: bool,
        language_model_only: bool,
        text_only_multimodal: bool,
    ) -> None:
        from transformers import AutoTokenizer  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore

        patch_vllm_transformers_base_for_nullable_subconfigs()
        runtime_model = prepare_model_dir_for_vllm(model)

        llm_kwargs: dict[str, Any] = {
            "model": runtime_model,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": enforce_eager,
        }
        if language_model_only:
            llm_kwargs["language_model_only"] = True
        if text_only_multimodal:
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}

        self._llm = LLM(
            **llm_kwargs,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(runtime_model, trust_remote_code=trust_remote_code)
        self._sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            structured_outputs=StructuredOutputsParams(json=ACTIVITY_CHAIN_JSON_SCHEMA),
        )

    def generate_batch(self, scenario_seeds: list[dict[str, Any]]) -> list[BackendResponse]:
        prompts = [build_chat_prompt(self._tokenizer, scenario_seed=seed) for seed in scenario_seeds]
        outputs = self._llm.generate(prompts, self._sampling)
        return [BackendResponse(raw_text=output.outputs[0].text.strip()) for output in outputs]


class TransformersBackend:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_model_len: int,
        dtype: str,
        seed: int,
        trust_remote_code: bool,
        device_map: str,
        attn_implementation: str | None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed  # type: ignore

        local_files_only = _is_local_model_path(model)
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        self._processor = AutoProcessor.from_pretrained(model, **processor_kwargs)
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        if getattr(self._tokenizer, "pad_token", None) is None and getattr(self._tokenizer, "eos_token", None) is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if hasattr(self._tokenizer, "padding_side"):
            self._tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_torch_dtype(dtype),
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        self._model.eval()
        self._input_device = _model_input_device(self._model)
        self._max_model_len = max_model_len
        self._generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self._tokenizer, "eos_token_id", None),
        }
        if temperature > 0.0:
            self._generation_kwargs["do_sample"] = True
            self._generation_kwargs["temperature"] = temperature
            self._generation_kwargs["top_p"] = top_p
        else:
            self._generation_kwargs["do_sample"] = False

        set_seed(seed)

    def generate_batch(self, scenario_seeds: list[dict[str, Any]]) -> list[BackendResponse]:
        import torch

        prompts: list[str] = []
        for scenario_seed in scenario_seeds:
            messages = build_processor_messages(scenario_seed)
            try:
                prompt = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            prompts.append(prompt)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_model_len,
        )
        inputs = inputs.to(self._input_device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **self._generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        decoded = self._tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [BackendResponse(raw_text=text.strip()) for text in decoded]


class MockBackend:
    def __init__(self, *, invalid_every: int = 4) -> None:
        self.invalid_every = invalid_every

    def generate_batch(self, scenario_seeds: list[dict[str, Any]]) -> list[BackendResponse]:
        responses: list[BackendResponse] = []
        for scenario_seed in scenario_seeds:
            record = _build_mock_record(scenario_seed)
            scenario_number = int(scenario_seed["scenario_id"].split("_")[-1])
            if self.invalid_every > 0 and (scenario_number + 1) % self.invalid_every == 0:
                record["summary"]["out_of_home_activities"] += 1
            responses.append(BackendResponse(raw_text=json.dumps(record, ensure_ascii=False)))
        return responses


def _scenario_number(scenario_seed: dict[str, Any]) -> int:
    return int(scenario_seed["scenario_id"].split("_")[-1])


def _choose_mode(scenario_seed: dict[str, Any], *, distance_km: float) -> tuple[str, list[str]]:
    persona = scenario_seed["persona"]
    environment = scenario_seed["environment"]
    reasons: list[str] = []

    if environment["weather"] in {"heavy_rain", "snow"}:
        reasons.append("weather")
    if environment["transit_disruption"] in {"major_delay", "partial_shutdown"}:
        reasons.append("transit_delay")
    if environment["parking_cost_level"] == "high":
        reasons.append("parking")
    if environment["toll_level"] == "high":
        reasons.append("toll")

    if persona["car_ownership"] != "none" and (
        environment["transit_disruption"] in {"major_delay", "partial_shutdown"}
        or environment["weather"] in {"heavy_rain", "snow"}
        or distance_km > 10.0
    ):
        return "drive", (reasons + ["comfort", "time"])[:3]

    if persona["bike_ownership"] and environment["weather"] == "clear" and distance_km <= 6.0 and persona["age"] < 60:
        return "bike", (reasons + ["time", "habit"])[:3]

    if persona["transit_pass"] and environment["transit_disruption"] not in {"major_delay", "partial_shutdown"}:
        return "transit", (reasons + ["cost", "reliability"])[:3]

    if distance_km <= 2.0:
        return "walk", (reasons + ["cost", "habit"])[:3]

    if persona["car_ownership"] != "none":
        return "drive", (reasons + ["time", "comfort"])[:3]

    return "ridehail", (reasons + ["time", "comfort"])[:3]


def _primary_activity(scenario_seed: dict[str, Any]) -> tuple[str, str]:
    persona = scenario_seed["persona"]
    environment = scenario_seed["environment"]
    day_type = environment["day_type"]
    employment = persona["employment_status"]

    if day_type in {"saturday", "sunday"}:
        options = [("leisure", "park_zone"), ("shopping", "retail_strip"), ("social", "friend_zone")]
        return options[_scenario_number(scenario_seed) % len(options)]

    if employment == "student" and environment["special_event"] != "school_holiday":
        return "education", "school_zone"
    if employment in {"full_time", "part_time"}:
        return ("remote_work", "office_core") if persona["work_schedule"] == "remote" else ("work", "office_core")
    if employment == "retired":
        return "medical", "medical_cluster"
    return "errand", "retail_strip"


def _mode_speed_kmh(mode: str) -> float:
    return {
        "walk": 4.8,
        "bike": 15.0,
        "transit": 28.0,
        "drive": 36.0,
        "ridehail": 32.0,
    }[mode]


def _travel_time_min(mode: str, *, distance_km: float) -> int:
    speed = _mode_speed_kmh(mode)
    return max(1, int(round((distance_km / speed) * 60.0)))


def _build_leg(
    *,
    sequence: int,
    from_activity_sequence: int,
    to_activity_sequence: int,
    depart_min: int,
    distance_km: float,
    scenario_seed: dict[str, Any],
) -> dict[str, Any]:
    mode, reasons = _choose_mode(scenario_seed, distance_km=distance_km)
    if not reasons:
        reasons = ["habit"]
    travel_time_min = _travel_time_min(mode, distance_km=distance_km)
    generalized_cost = round(distance_km * {"walk": 0.0, "bike": 0.2, "transit": 0.8, "drive": 1.3, "ridehail": 2.0}[mode], 2)
    return {
        "sequence": sequence,
        "from_activity_sequence": from_activity_sequence,
        "to_activity_sequence": to_activity_sequence,
        "mode": mode,
        "depart_min": depart_min,
        "arrive_min": depart_min + travel_time_min,
        "travel_time_min": travel_time_min,
        "distance_km": round(distance_km, 1),
        "generalized_cost": generalized_cost,
        "reason_tags": list(dict.fromkeys([reason for reason in reasons if reason in REASON_TAGS]))[:3] or ["habit"],
    }


def _build_mock_record(scenario_seed: dict[str, Any]) -> dict[str, Any]:
    scenario_number = _scenario_number(scenario_seed)
    primary_purpose, primary_zone = _primary_activity(scenario_seed)
    weekday = scenario_seed["environment"]["day_type"] in {"weekday", "friday"}

    sprawl = scenario_seed.get("macro_context", {}).get("spatial_sprawl_score", 0.5)
    distance_scale = 0.6 + sprawl * 0.8

    depart_home = 420 + (scenario_number % 8) * 15 if weekday else 540 + (scenario_number % 8) * 20
    outbound_distance = round((2.5 + (scenario_number % 7) * 1.3) * distance_scale, 1)
    outbound_leg = _build_leg(
        sequence=0,
        from_activity_sequence=0,
        to_activity_sequence=1,
        depart_min=depart_home,
        distance_km=outbound_distance,
        scenario_seed=scenario_seed,
    )
    primary_start = outbound_leg["arrive_min"]

    if primary_purpose in {"work", "education"}:
        primary_duration = 420 if weekday else 180
    elif primary_purpose == "remote_work":
        primary_duration = 120
    else:
        primary_duration = 90 + (scenario_number % 4) * 30
    primary_end = primary_start + primary_duration

    activities = [
        {
            "sequence": 0,
            "purpose": "home",
            "location_zone": "home_zone",
            "start_min": 0,
            "end_min": depart_home,
            "flexibility": "low",
        },
        {
            "sequence": 1,
            "purpose": primary_purpose,
            "location_zone": primary_zone,
            "start_min": primary_start,
            "end_min": primary_end,
            "flexibility": "low" if primary_purpose in {"work", "education"} else "medium",
        },
    ]
    legs = [outbound_leg]

    add_secondary = scenario_number % 2 == 0
    if add_secondary:
        secondary_purpose = "shopping" if scenario_number % 4 == 0 else "meal"
        secondary_zone = "retail_strip" if secondary_purpose == "shopping" else ZONE_CHOICES[(scenario_number + 3) % len(ZONE_CHOICES)]
        secondary_distance = round((1.2 + (scenario_number % 5) * 0.8) * distance_scale, 1)
        leg_to_secondary = _build_leg(
            sequence=1,
            from_activity_sequence=1,
            to_activity_sequence=2,
            depart_min=primary_end,
            distance_km=secondary_distance,
            scenario_seed=scenario_seed,
        )
        secondary_start = leg_to_secondary["arrive_min"]
        secondary_end = secondary_start + 45 + (scenario_number % 3) * 15
        activities.append(
            {
                "sequence": 2,
                "purpose": secondary_purpose,
                "location_zone": secondary_zone,
                "start_min": secondary_start,
                "end_min": secondary_end,
                "flexibility": "medium",
            }
        )
        legs.append(leg_to_secondary)
        return_depart = secondary_end
        return_from_sequence = 2
        home_sequence = 3
    else:
        return_depart = primary_end
        return_from_sequence = 1
        home_sequence = 2

    return_distance = round(outbound_distance + 0.7 * distance_scale, 1)
    return_leg = _build_leg(
        sequence=len(legs),
        from_activity_sequence=return_from_sequence,
        to_activity_sequence=home_sequence,
        depart_min=return_depart,
        distance_km=return_distance,
        scenario_seed=scenario_seed,
    )
    home_start = return_leg["arrive_min"]
    activities.append(
        {
            "sequence": home_sequence,
            "purpose": "home",
            "location_zone": "home_zone",
            "start_min": home_start,
            "end_min": 1440,
            "flexibility": "high",
        }
    )
    legs.append(return_leg)

    total_distance = round(sum(float(leg["distance_km"]) for leg in legs), 1)
    total_travel_time = sum(int(leg["travel_time_min"]) for leg in legs)
    commute_mode = legs[0]["mode"] if primary_purpose in {"work", "education", "remote_work"} else "none"
    out_of_home = sum(1 for activity in activities if activity["purpose"] != "home")

    return {
        "schema_version": "activity_chain_v1",
        "scenario_id": scenario_seed["scenario_id"],
        "city_archetype": scenario_seed["city_archetype"],
        "macro_context": scenario_seed["macro_context"],
        "persona": scenario_seed["persona"],
        "environment": scenario_seed["environment"],
        "activities": activities,
        "legs": legs,
        "summary": {
            "primary_tour_purpose": primary_purpose,
            "commute_mode_if_any": commute_mode,
            "out_of_home_activities": out_of_home,
            "total_distance_km": total_distance,
            "total_travel_time_min": total_travel_time,
        },
    }
