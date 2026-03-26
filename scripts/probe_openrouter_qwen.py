from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activity_chain.prompting import build_user_prompt
from activity_chain.schema import ACTIVITY_CHAIN_JSON_SCHEMA, SYSTEM_PROMPT
from activity_chain.seeds import iter_scenario_seeds


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3.5-9B on OpenRouter for prompt/schema exploration."
    )
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--site-url", default=os.environ.get("OPENROUTER_SITE_URL", "https://bonzai.local"))
    parser.add_argument("--app-name", default=os.environ.get("OPENROUTER_APP_NAME", "Bonzai SAT Local Probe"))
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--user-prompt", default=None)
    parser.add_argument("--user-prompt-file", default=None)
    parser.add_argument("--seed-file", default=None, help="Scenario seed JSON file to build the current Bonzai prompt.")
    parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help="Generate a Bonzai scenario seed locally and use that seed index as the prompt input.",
    )
    parser.add_argument(
        "--seed-rng",
        type=int,
        default=7,
        help="Random seed used when generating a local Bonzai scenario seed via --seed-index.",
    )
    parser.add_argument(
        "--schema-file",
        default=None,
        help="Optional JSON schema file for structured outputs. Mutually compatible with --use-activity-schema.",
    )
    parser.add_argument(
        "--use-activity-schema",
        action="store_true",
        help="Use the current ACTIVITY_CHAIN_JSON_SCHEMA for structured output probing.",
    )
    parser.add_argument(
        "--schema-name",
        default="bonzai_probe",
        help="Name field used for OpenRouter structured outputs.",
    )
    parser.add_argument(
        "--dump-response",
        default=None,
        help="Optional path to save the full raw OpenRouter JSON response.",
    )
    parser.add_argument(
        "--dump-content",
        default=None,
        help="Optional path to save only the assistant content.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider routing JSON string, passed through to OpenRouter.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --api-key.")
    if args.schema_file and args.use_activity_schema:
        raise SystemExit("Use either --schema-file or --use-activity-schema, not both.")
    seed_input_count = sum(
        value is not None
        for value in [
            args.seed_file,
            args.seed_index,
        ]
    )
    if seed_input_count > 1:
        raise SystemExit("Use only one of --seed-file or --seed-index.")
    if (args.seed_file or args.seed_index is not None) and (args.user_prompt or args.user_prompt_file):
        raise SystemExit("Use either a seed input or an explicit user prompt input, not both.")
    return args


def _read_text(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _load_seed_prompt(seed_file: str) -> tuple[str, str]:
    scenario_seed = json.loads(Path(seed_file).read_text(encoding="utf-8"))
    return SYSTEM_PROMPT, build_user_prompt(scenario_seed)


def _generate_seed_prompt(seed_index: int, seed_rng: int) -> tuple[str, str]:
    if seed_index < 0:
        raise SystemExit("--seed-index must be >= 0.")
    iterator = iter_scenario_seeds(total=seed_index + 1, seed=seed_rng)
    scenario_seed: dict[str, Any] | None = None
    for scenario_seed in iterator:
        pass
    if scenario_seed is None:
        raise SystemExit("Failed to generate a scenario seed.")
    return SYSTEM_PROMPT, build_user_prompt(scenario_seed)


def _resolve_prompts(args: argparse.Namespace) -> tuple[str, str]:
    if args.seed_file:
        return _load_seed_prompt(args.seed_file)
    if args.seed_index is not None:
        return _generate_seed_prompt(args.seed_index, args.seed_rng)

    system_prompt = args.system_prompt or _read_text(args.system_prompt_file) or "You are a helpful assistant."
    user_prompt = args.user_prompt or _read_text(args.user_prompt_file)
    if not user_prompt:
        raise SystemExit("Provide --user-prompt, --user-prompt-file, --seed-file, or --seed-index.")
    return system_prompt, user_prompt


def _load_schema(args: argparse.Namespace) -> dict[str, Any] | None:
    schema: dict[str, Any] | None = None
    if args.use_activity_schema:
        schema = ACTIVITY_CHAIN_JSON_SCHEMA
    elif args.schema_file:
        schema = json.loads(Path(args.schema_file).read_text(encoding="utf-8"))
    return schema


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    system_prompt, user_prompt = _resolve_prompts(args)

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    schema = _load_schema(args)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": args.schema_name,
                "strict": True,
                "schema": schema,
            },
        }

    if args.provider:
        payload["provider"] = json.loads(args.provider)

    return payload


def _post_json(*, api_key: str, site_url: str, app_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise SystemExit(f"OpenRouter request failed: {exc}") from exc


def _extract_content(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        raise SystemExit("OpenRouter response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _dump_if_requested(path: str | None, content: str) -> None:
    if path:
        Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    payload = _build_payload(args)
    response = _post_json(
        api_key=args.api_key,
        site_url=args.site_url,
        app_name=args.app_name,
        payload=payload,
    )

    content = _extract_content(response)
    usage = response.get("usage")

    print("=== Request Summary ===")
    print(f"model={args.model}")
    print(f"temperature={args.temperature}")
    print(f"max_tokens={args.max_tokens}")
    print(f"structured_output={'response_format' in payload}")
    print()

    if usage:
        print("=== Usage ===")
        print(json.dumps(usage, indent=2, ensure_ascii=False))
        print()

    print("=== Assistant Content ===")
    print(content)

    _dump_if_requested(args.dump-response, json.dumps(response, indent=2, ensure_ascii=False))
    _dump_if_requested(args.dump-content, content)


if __name__ == "__main__":
    main()
