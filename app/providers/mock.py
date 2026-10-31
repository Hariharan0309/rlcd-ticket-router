"""A deterministic keyword-based stand-in for a real model.

It lets the page and the tests run with no API key and no model download.
It is NOT smart: it only counts keywords, which is exactly why it is useful
as a contrast to the real models on vague and tricky tickets.
"""

import math
import re

from app.providers.base import TEAMS, Answer, Provider

KEYWORDS: dict[str, list[str]] = {
    "billing": ["charge", "charged", "invoice", "payment", "billing", "refund", "price", "card"],
    "technical": ["bug", "error", "crash", "crashes", "broken", "down", "not working", "export", "update"],
    "sales": ["buy", "upgrade", "quote", "demo", "pricing", "seats", "plan", "enterprise"],
    "account": ["login", "log in", "password", "sign in", "locked", "profile", "username", "2fa"],
}
URGENT_WORDS = ["urgent", "asap", "immediately", "now", "today", "can't work", "cannot work", "outage", "down"]
REFUND_WORDS = ["refund", "money back", "chargeback", "reimburse"]


def _count(text: str, words: list[str]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", text)) for w in words)


class MockProvider(Provider):
    name = "mock"

    def _answer(self, ticket_text: str) -> Answer:
        text = ticket_text.lower()

        # Each keyword hit adds 2 to a team's score; "other" gets a small base score
        # so a ticket with no keywords lands on "other". Softmax turns scores into
        # probabilities, so more hits = more confident.
        scores = {team: 2.0 * _count(text, KEYWORDS[team]) for team in KEYWORDS}
        scores["other"] = 0.5
        exps = {team: math.exp(scores[team]) for team in TEAMS}
        total = sum(exps.values())
        team_probs = {team: round(exps[team] / total, 4) for team in TEAMS}

        urgency = min(5.0, 2.0 + _count(text, URGENT_WORDS) + min(text.count("!"), 1))
        refund_prob = 0.95 if _count(text, REFUND_WORDS) else 0.05
        return Answer(team_probs, urgency, refund_prob)
