"""Jev (TypeSafe AI), called directly over HTTP at api.typesafe.ai.

API reference: https://docs.typesafe.ai/api/system-one.md
Real output from scripts/probe_jev.py (trimmed):
  {"model": "jev-1.14.0",
   "answers": {
      "team":             {"type": "choice", "choice": "billing", "confidence": 1.0,
                           "probabilities": {"billing": 1.0, ...}},
      "urgency":          {"type": "score", "score": 2.05, "confidence": 0.8,
                           "legend": {"0": "1: no rush", ...}, "probabilities": {"0": 0.0, "1": 0.09, ...}},
      "refund_requested": {"type": "noul", "noul": 0.99}},
   "usage": {"input_tokens": 346, "output_tokens": 38}}
"""

import os

import httpx

from app.providers.base import (
    REFUND_INSTRUCTIONS,
    TEAM_INSTRUCTIONS,
    TEAMS,
    URGENCY_INSTRUCTIONS,
    URGENCY_LEVELS,
    Answer,
    Provider,
    ProviderError,
)

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"  # alias; the response names the exact version that answered
TIMEOUT_S = 15.0

QUESTIONS: dict[str, dict] = {
    "team": {"type": "choice", "instructions": TEAM_INSTRUCTIONS, "criteria": TEAMS},
    "urgency": {"type": "score", "instructions": URGENCY_INSTRUCTIONS, "criteria": URGENCY_LEVELS},
    "refund_requested": {"type": "noul", "instructions": REFUND_INSTRUCTIONS},  # noul = yes/no
}

# Plain-English hints for the status codes TypeSafe documents.
STATUS_HINTS: dict[int, str] = {
    401: "the API key is missing or invalid",
    422: "the request was rejected as invalid",
    429: "rate limit reached, try again in a moment",
    529: "TypeSafe is temporarily overloaded, try again in a moment",
}


class JevProvider(Provider):
    name = "jev"
    # Listed price for Jev 1.13/1.14, input tokens only (output is free).
    # Source: https://docs.typesafe.ai/models.md, checked 2026-10-30. Update if it changes.
    price_per_mtok = 0.042
    runs_on = "TypeSafe cloud"

    def __init__(self) -> None:
        # One client for the app's lifetime: reusing the TLS connection keeps
        # latency_ms about the model, not about handshakes.
        self._client = httpx.Client(timeout=TIMEOUT_S)

    def _answer(self, ticket_text: str) -> Answer:
        key = os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise ProviderError("No TYPESAFE_API_KEY in .env, so Jev can't be called.")

        body = {"model": MODEL, "state": ticket_text, "questions": QUESTIONS}
        try:
            resp = self._client.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        except httpx.TimeoutException:
            raise ProviderError(f"Jev did not answer within {TIMEOUT_S:.0f}s.")
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not reach TypeSafe ({type(e).__name__}).")

        if resp.status_code != 200:
            hint = STATUS_HINTS.get(resp.status_code, _error_message(resp))
            raise ProviderError(f"TypeSafe returned HTTP {resp.status_code}: {hint}")
        return parse_response(resp.json())


def parse_response(data: dict) -> Answer:
    """Turn Jev's JSON into an Answer (urgency on a 1-5 scale)."""
    try:
        answers = data["answers"]
        probs = answers["team"]["probabilities"]
        # Missing options count as 0 so the bars always show all five teams.
        team_probs = {team: float(probs.get(team, 0.0)) for team in TEAMS}
        # Jev scores levels starting at 0 (see "legend"), so +1 maps our 5 labels onto 1 to 5.
        urgency = float(answers["urgency"]["score"]) + 1
        refund_prob = float(answers["refund_requested"]["noul"])
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ProviderError(f"Unexpected Jev response format (missing {e}).")
    input_tokens = (data.get("usage") or {}).get("input_tokens")
    return Answer(team_probs, urgency, refund_prob, input_tokens)


def _error_message(resp: httpx.Response) -> str:
    """Pull a readable message out of an error reply, whatever its shape."""
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:200] or "no details"
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        return str(err.get("message") or err.get("detail") or err)[:200]
    return str(err)[:200]
