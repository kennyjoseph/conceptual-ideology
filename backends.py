"""LLM backends for the hypothesis pipeline.

Everything here exists to turn a prompt plus a JSON schema into a dict. Two
implementations: the Anthropic API, which honors a schema natively, and anything
that speaks the OpenAI chat-completions API (the UB BullsAI gateway, Gemini's
compat endpoint, a local vLLM server, ...), which mostly doesn't.

Pick one with make_backend(args); it reads --backend, --model, --base-url and
--api-key off the parsed CLI namespace.
"""

import argparse
import json
import os
import sys
from typing import Protocol

import anthropic
import openai

ANTHROPIC_MODEL = "claude-haiku-4-5"

# The OpenAI-compatible gateway this project was run against. The URL is only a default
# and `--base-url` overrides it; the key is never stored here. Set GATEWAY_API_KEY in the
# environment, or pass `--api-key`, and `gateway_key()` will find it.
GATEWAY_BASE_URL = os.environ.get(
    "GATEWAY_BASE_URL", "https://gateway.bullsai.buffalo.edu/v1"
)
GATEWAY_MODEL = "zai-org/GLM-4.7-Flash"


def gateway_key(explicit: str | None = None) -> str:
    """The gateway credential, from `--api-key` or the environment.

    Raises rather than falling back to a default, because the alternative -- a key sitting
    in the source -- is how this file previously leaked one.
    """
    key = explicit or os.environ.get("GATEWAY_API_KEY")
    if not key:
        raise SystemExit(
            "no gateway credential: set GATEWAY_API_KEY in the environment or pass "
            "--api-key. (The extraction step needs this; model fitting does not.)"
        )
    return key


# Text-capable chat models on the gateway, for reference when picking --model.
GATEWAY_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Qwen/Qwen3.6-27B",
    "google/gemma-4-31B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-3-27b-it",
    "mistralai/Ministral-3-14B-Instruct",
    "zai-org/GLM-5.2",
    "zai-org/GLM-4.7-Flash",
]

# Seconds to wait for a reply. Big models on a busy gateway can be slow.
TIMEOUT = 180

# 0 means a hung request fails right away instead of quietly taking 3 x TIMEOUT.
MAX_RETRIES = 0


class LLMBackend(Protocol):
    """Anything that can turn a prompt into a JSON object matching a schema."""

    def complete(self, user_message: str, schema: dict, max_tokens: int) -> dict: ...


def parse_json_response(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Models that honor a JSON schema return bare JSON and this is just json.loads.
    Models that don't (most open models on the gateway) wrap it in ``` fences or
    preface it with a sentence of prose, so fall back to grabbing the outermost
    braces.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:300]!r}")
    return json.loads(text[start : end + 1])


class AnthropicBackend:
    """Anthropic API with native structured output."""

    def __init__(self, model: str = ANTHROPIC_MODEL, api_key: str | None = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, user_message: str, schema: dict, max_tokens: int) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user_message}],
        )
        return parse_json_response(
            next(b.text for b in response.content if b.type == "text")
        )


class OpenAICompatBackend:
    """Any endpoint that speaks the OpenAI chat-completions API.

    Tries `response_format` with a JSON schema first; a lot of models served
    through the gateway reject it, so on failure we retry once without it and
    lean on parse_json_response plus the "return only JSON" instruction in the
    prompt.
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        self.supports_schema = True

    @staticmethod
    def _content(response) -> str:
        """Pull the text out, with a readable error when there isn't any.

        Reasoning models on the gateway (GLM, gpt-oss) spend the token budget on
        hidden reasoning and then return nothing, which otherwise surfaces as a
        confusing "no JSON found in ''".
        """
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise ValueError(
                f"Empty response (finish_reason={choice.finish_reason!r}). "
                "If this is 'length', the model burned the budget on reasoning "
                "tokens -- raise --max-tokens."
            )
        return content

    def complete(self, user_message: str, schema: dict, max_tokens: int) -> dict:
        messages = [{"role": "user", "content": user_message}]

        if self.supports_schema:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                )
                return parse_json_response(self._content(response))
            except openai.BadRequestError as e:
                print(
                    f"  ({self.model} rejected json_schema: {e}; falling back to plain text)",
                    file=sys.stderr,
                )
                self.supports_schema = False

        response = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=messages
        )
        return parse_json_response(self._content(response))


def make_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "anthropic":
        return AnthropicBackend(
            model=args.model or ANTHROPIC_MODEL, api_key=args.api_key
        )
    if args.backend == "gateway":
        return OpenAICompatBackend(
            model=args.model or GATEWAY_MODEL,
            base_url=args.base_url or GATEWAY_BASE_URL,
            api_key=gateway_key(args.api_key),
        )
    # openai-compat: everything must be given explicitly.
    if not args.base_url or not args.model:
        raise SystemExit("--backend openai-compat requires --base-url and --model")
    return OpenAICompatBackend(
        model=args.model, base_url=args.base_url, api_key=args.api_key or "EMPTY"
    )
