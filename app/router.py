"""The business rule: when is a model's answer good enough to act on alone?

This is the whole point of the demo. The model gives probabilities; *we* pick
the threshold. Raise it and more tickets go to a human; lower it and more get
auto-handled. (The page re-runs this same rule in JavaScript so the slider
can move the badge instantly. Keep the two in sync.)
"""

import math

from pydantic import BaseModel

from app.providers.base import Decision

AUTO_HANDLE = "AUTO-HANDLE"
SEND_TO_HUMAN = "SEND TO HUMAN"
URGENT_AT = 4  # urgency >= 4 gets an "urgent" flag


class Routing(BaseModel):
    action: str
    team: str
    confidence: float
    urgent: bool
    reason: str


def pct(p: float) -> str:
    # Round down, so 0.8996 shows as 89% and never looks like it passed a 90% threshold.
    return f"{math.floor(p * 100 + 1e-9)}%"


def route(decision: Decision, threshold: float) -> Routing:
    team, confidence = decision.top_team, decision.confidence
    label = f"{team.capitalize()} at {pct(confidence)}"

    # "other" means the model thinks no team fits, so a human must look,
    # however confident it is about that.
    if team == "other":
        action, reason = SEND_TO_HUMAN, f"{label}: no team fits, a human should look"
    elif confidence >= threshold:
        action, reason = AUTO_HANDLE, f"{label}, above the {pct(threshold)} threshold"
    else:
        action, reason = SEND_TO_HUMAN, f"{label}, below the {pct(threshold)} threshold"

    return Routing(
        action=action,
        team=team,
        confidence=confidence,
        urgent=decision.urgency >= URGENT_AT,
        reason=reason,
    )


def unavailable(provider: str) -> Routing:
    """No answer from the model means no confidence, so a human handles it."""
    return Routing(
        action=SEND_TO_HUMAN,
        team="unknown",
        confidence=0.0,
        urgent=False,
        reason=f"{provider.capitalize()} gave no answer, so a human should look",
    )
