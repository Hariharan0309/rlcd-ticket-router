"""The shared contract every model provider follows.

Both real models get the exact same three questions (wording, options, scale),
so differences on screen come from the models, not from the prompts.
"""

import time
from abc import ABC, abstractmethod
from typing import NamedTuple

from pydantic import BaseModel

# Question 1: team (choice). Each option has a short description because both
# Jev and Laya take choice options as {label: description}.
TEAMS: dict[str, str] = {
    "billing": "charges, invoices, payments, refunds, pricing of an existing plan",
    "technical": "bugs, errors, crashes, outages, a feature not working",
    "sales": "buying, upgrading, quotes, demos, new plans or seats",
    "account": "login, password, sign-in, profile, user access, closing the account",
    "other": "anything that is not a support request for the teams above",
}
TEAM_INSTRUCTIONS = "Which support team should handle this ticket?"

# Question 2: urgency (score). Ordered lowest to highest; we report it as 1 to 5.
URGENCY_LEVELS: list[str] = [
    "1: no rush, general question",
    "2: low, minor inconvenience",
    "3: normal, should be handled today",
    "4: high, blocking the customer's work",
    "5: critical, outage or money at risk right now",
]
URGENCY_INSTRUCTIONS = "How urgent is this ticket?"

# Question 3: refund_requested (yes/no probability).
REFUND_INSTRUCTIONS = "Is the customer asking for a refund or their money back?"


class Answer(NamedTuple):
    """What a provider's parser returns: the three answers plus token usage."""

    team_probs: dict[str, float]
    urgency: float
    refund_prob: float
    input_tokens: int | None = None  # as reported by the model; None for the mock


class Decision(BaseModel):
    """What one model said about one ticket."""

    provider: str
    team_probs: dict[str, float]  # one probability per team, sums to ~1
    urgency: float  # 1.0 to 5.0 (models may return a fractional expected level)
    refund_prob: float  # P(refund requested), 0.0 to 1.0
    latency_ms: float  # wall-clock time of the model call only
    input_tokens: int | None = None
    cost_usd: float | None = None  # input_tokens x the listed price; 0 for local models
    runs_on: str | None = None  # where the model ran, shown on the card

    @property
    def top_team(self) -> str:
        return max(self.team_probs, key=self.team_probs.__getitem__)

    @property
    def confidence(self) -> float:
        return self.team_probs[self.top_team]


class ProviderError(Exception):
    """A failure we can explain to the viewer (missing key, timeout, bad reply)."""


class Provider(ABC):
    name: str = "provider"
    price_per_mtok: float = 0.0  # USD per million input tokens; 0 for models that run locally
    runs_on: str | None = None  # e.g. "TypeSafe cloud", "your GPU"

    def warm_up(self) -> None:
        """Load anything slow (e.g. model weights). Not counted in latency_ms."""

    @abstractmethod
    def _answer(self, ticket_text: str) -> Answer:
        """Return the answers (urgency on a 1-5 scale). Raise ProviderError on failure."""

    def decide(self, ticket_text: str) -> Decision:
        self.warm_up()
        # Time only the call itself so the number on screen is the real per-ticket cost.
        start = time.perf_counter()
        answer = self._answer(ticket_text)
        latency_ms = (time.perf_counter() - start) * 1000
        cost = None if answer.input_tokens is None else answer.input_tokens * self.price_per_mtok / 1e6
        return Decision(
            provider=self.name,
            team_probs=answer.team_probs,
            urgency=answer.urgency,
            refund_prob=answer.refund_prob,
            latency_ms=round(latency_ms, 1),
            input_tokens=answer.input_tokens,
            cost_usd=cost,
            runs_on=self.runs_on,
        )
