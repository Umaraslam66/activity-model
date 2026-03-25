from __future__ import annotations

from typing import Any


CITY_ARCHETYPES = ["dense_transit_hub", "sprawling_car_city", "winter_cycling_city"]

SYSTEM_PROMPT = """You generate exactly one synthetic urban mobility day-plan for an activity-based model.
Return exactly one JSON object. No markdown. No commentary. No chain-of-thought.

Your job is to produce a physically plausible 24-hour activity chain for one synthetic person living in a city.
You will be given a fixed persona seed, a fixed environment seed, a city_archetype, and a macro_context block. Preserve all seed attributes exactly — copy city_archetype and macro_context verbatim into your output.

City archetypes:
- "dense_transit_hub": compact city with excellent public transit, short distances, low car dependency. Transit and walking should dominate mode share. Driving is costly and slow.
- "sprawling_car_city": spread-out metro area with limited transit, long distances, high car dependency. Driving dominates. Walking/cycling only viable for very short trips.
- "winter_cycling_city": Nordic-style city with strong cycling infrastructure even in cold weather. Cycling is viable year-round for resilient personas. Transit is a strong fallback.

Macro context fields:
- transit_density_index (0-1): higher means better transit coverage and frequency. Should increase transit mode share.
- active_mobility_index (0-1): higher means better bike/walk infrastructure. Should increase cycling/walking for short-medium trips.
- spatial_sprawl_score (0-1): higher means more sprawl, longer average distances between activity locations.

Latent persona traits:
- value_of_time_multiplier (0.5-2.0): higher means the persona values time savings more — will pay more for faster modes.
- weather_resilience (0-1): higher means less deterred by bad weather for active modes (walking, cycling).
- comfort_preference (0-1): higher means stronger preference for comfortable modes (car, ridehail) over cheaper/greener ones.

Micro context shocks:
- dynamic_cost_sek (0-500): real-time cost shock in SEK. High values should push personas toward cheaper modes or reduce optional trips.
- network_delay_mins (0-60): real-time network disruption in minutes. High values should reduce transit attractiveness and may trigger mode switching.

Use these fields to inform mode choice, trip timing, and activity scheduling. A persona with high weather_resilience in a winter_cycling_city may still bike in snow. A persona with high value_of_time_multiplier in a sprawling_car_city will strongly prefer driving. High dynamic_cost_sek should discourage ridehail and tolled driving. High network_delay_mins should discourage transit.

Hard constraints:
- The plan must cover one full day from minute 0 to minute 1440.
- The first activity must start at 0 and the last activity must end at 1440.
- Activities and travel legs must be strictly chronological with no overlaps.
- Each leg must connect activity i to activity i+1.
- leg.depart_min must equal the previous activity end_min.
- leg.arrive_min must equal the next activity start_min.
- leg.travel_time_min must equal arrive_min - depart_min.
- No teleportation.
- If car_ownership is "none", do not use mode "drive".
- If bike_ownership is false, do not use mode "bike".
- Weather, transit disruption, toll level, congestion, and parking cost must visibly influence mode and timing choices when relevant.
- Keep the plan behaviorally realistic for the person's age, employment status, household structure, and work schedule.

Physical plausibility:
- walk speed should usually stay between 3 and 7 km/h
- bike speed should usually stay between 8 and 25 km/h
- transit effective speed should usually stay between 10 and 70 km/h
- drive effective speed should usually stay between 10 and 120 km/h
- ridehail effective speed should usually stay between 10 and 120 km/h
- Avoid impossible long-distance trips in very short times.

Behavioral plausibility:
- Most weekday workers should show a work or remote-work pattern.
- Students should show education-related travel on weekdays unless the environment makes that implausible.
- Retired or unemployed personas may still perform errands, social, medical, exercise, shopping, and leisure activities.
- Households with children may include escort or pickup/dropoff activities.
- Bad weather should reduce walking/biking propensity for borderline trips.
- Major transit disruption should reduce transit usage unless the traveler lacks good alternatives.
- High tolls and high parking cost should reduce driving attractiveness when substitutes exist.

Style constraints:
- Use compact categorical fields only.
- Do not invent free-text explanations.
- reason_tags must be short labels chosen from the allowed enum.
- Use realistic zone names like "home_zone", "office_core", "school_zone", "retail_strip", "gym_district", "medical_cluster", "park_zone", "friend_zone", "transit_hub".
"""


ACTIVITY_PURPOSES = [
    "home",
    "work",
    "education",
    "shopping",
    "escort",
    "leisure",
    "meal",
    "errand",
    "medical",
    "pickup_dropoff",
    "social",
    "exercise",
    "remote_work",
    "other",
]

MODE_CHOICES = ["walk", "bike", "transit", "drive", "ridehail"]
FLEXIBILITY_CHOICES = ["low", "medium", "high"]
REASON_TAGS = [
    "habit",
    "cost",
    "time",
    "comfort",
    "reliability",
    "weather",
    "safety",
    "childcare",
    "parking",
    "toll",
    "transit_delay",
]

ZONE_CHOICES = [
    "home_zone",
    "office_core",
    "school_zone",
    "retail_strip",
    "gym_district",
    "medical_cluster",
    "park_zone",
    "friend_zone",
    "transit_hub",
    "university_quarter",
    "civic_center",
]

PERSONA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "age",
        "income_bracket",
        "employment_status",
        "household_structure",
        "has_children",
        "car_ownership",
        "bike_ownership",
        "transit_pass",
        "work_schedule",
        "value_of_time_multiplier",
        "weather_resilience",
        "comfort_preference",
    ],
    "properties": {
        "age": {"type": "integer", "minimum": 18, "maximum": 90},
        "income_bracket": {
            "type": "string",
            "enum": ["low", "lower_middle", "upper_middle", "high"],
        },
        "employment_status": {
            "type": "string",
            "enum": ["full_time", "part_time", "student", "retired", "unemployed"],
        },
        "household_structure": {
            "type": "string",
            "enum": ["single", "couple", "family", "shared"],
        },
        "has_children": {"type": "boolean"},
        "car_ownership": {
            "type": "string",
            "enum": ["none", "shared_household", "dedicated"],
        },
        "bike_ownership": {"type": "boolean"},
        "transit_pass": {"type": "boolean"},
        "work_schedule": {
            "type": "string",
            "enum": [
                "fixed_day",
                "shift",
                "flexible",
                "student_day",
                "retired",
                "unemployed",
                "remote",
            ],
        },
        "value_of_time_multiplier": {"type": "number", "minimum": 0.5, "maximum": 2.0},
        "weather_resilience": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "comfort_preference": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

ENVIRONMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "day_type",
        "weather",
        "transit_disruption",
        "road_congestion",
        "parking_cost_level",
        "toll_level",
        "special_event",
        "dynamic_cost_sek",
        "network_delay_mins",
    ],
    "properties": {
        "day_type": {
            "type": "string",
            "enum": ["weekday", "friday", "saturday", "sunday"],
        },
        "weather": {
            "type": "string",
            "enum": ["clear", "rain", "heavy_rain", "snow", "heatwave"],
        },
        "transit_disruption": {
            "type": "string",
            "enum": ["none", "minor_delay", "major_delay", "partial_shutdown"],
        },
        "road_congestion": {
            "type": "string",
            "enum": ["light", "moderate", "heavy", "severe"],
        },
        "parking_cost_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "toll_level": {
            "type": "string",
            "enum": ["none", "moderate", "high"],
        },
        "special_event": {
            "type": "string",
            "enum": ["none", "sports_event", "concert", "street_festival", "school_holiday"],
        },
        "dynamic_cost_sek": {"type": "number", "minimum": 0.0, "maximum": 500.0},
        "network_delay_mins": {"type": "number", "minimum": 0.0, "maximum": 60.0},
    },
}

MACRO_CONTEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "transit_density_index",
        "active_mobility_index",
        "spatial_sprawl_score",
    ],
    "properties": {
        "transit_density_index": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "active_mobility_index": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "spatial_sprawl_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

ACTIVITY_CHAIN_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Activity chain day record",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "scenario_id",
        "city_archetype",
        "macro_context",
        "persona",
        "environment",
        "activities",
        "legs",
        "summary",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "activity_chain_v1"},
        "scenario_id": {"type": "string", "minLength": 1},
        "city_archetype": {"type": "string", "enum": CITY_ARCHETYPES},
        "macro_context": MACRO_CONTEXT_SCHEMA,
        "persona": PERSONA_SCHEMA,
        "environment": ENVIRONMENT_SCHEMA,
        "activities": {
            "type": "array",
            "minItems": 2,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sequence",
                    "purpose",
                    "location_zone",
                    "start_min",
                    "end_min",
                    "flexibility",
                ],
                "properties": {
                    "sequence": {"type": "integer", "minimum": 0, "maximum": 11},
                    "purpose": {"type": "string", "enum": ACTIVITY_PURPOSES},
                    "location_zone": {"type": "string", "minLength": 1},
                    "start_min": {"type": "integer", "minimum": 0, "maximum": 1439},
                    "end_min": {"type": "integer", "minimum": 1, "maximum": 1440},
                    "flexibility": {"type": "string", "enum": FLEXIBILITY_CHOICES},
                },
            },
        },
        "legs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 11,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sequence",
                    "from_activity_sequence",
                    "to_activity_sequence",
                    "mode",
                    "depart_min",
                    "arrive_min",
                    "travel_time_min",
                    "distance_km",
                    "generalized_cost",
                    "reason_tags",
                ],
                "properties": {
                    "sequence": {"type": "integer", "minimum": 0, "maximum": 10},
                    "from_activity_sequence": {"type": "integer", "minimum": 0, "maximum": 10},
                    "to_activity_sequence": {"type": "integer", "minimum": 1, "maximum": 11},
                    "mode": {"type": "string", "enum": MODE_CHOICES},
                    "depart_min": {"type": "integer", "minimum": 0, "maximum": 1439},
                    "arrive_min": {"type": "integer", "minimum": 1, "maximum": 1440},
                    "travel_time_min": {"type": "integer", "minimum": 1, "maximum": 300},
                    "distance_km": {"type": "number", "minimum": 0.1, "maximum": 120.0},
                    "generalized_cost": {"type": "number", "minimum": 0.0, "maximum": 250.0},
                    "reason_tags": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string", "enum": REASON_TAGS},
                    },
                },
            },
        },
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "primary_tour_purpose",
                "commute_mode_if_any",
                "out_of_home_activities",
                "total_distance_km",
                "total_travel_time_min",
            ],
            "properties": {
                "primary_tour_purpose": {"type": "string", "enum": ACTIVITY_PURPOSES},
                "commute_mode_if_any": {"type": "string", "enum": MODE_CHOICES + ["none"]},
                "out_of_home_activities": {"type": "integer", "minimum": 0, "maximum": 10},
                "total_distance_km": {"type": "number", "minimum": 0.0, "maximum": 400.0},
                "total_travel_time_min": {"type": "integer", "minimum": 0, "maximum": 600},
            },
        },
    },
}

MODE_SPEED_LIMITS_KMH = {
    "walk": (3.0, 7.0),
    "bike": (8.0, 25.0),
    "transit": (10.0, 70.0),
    "drive": (10.0, 120.0),
    "ridehail": (10.0, 120.0),
}
