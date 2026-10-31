from app.providers.base import TEAMS, Decision
from app.router import AUTO_HANDLE, SEND_TO_HUMAN, route, unavailable


def make_decision(top: str, p: float, urgency: float = 2.0) -> Decision:
    rest = (1 - p) / (len(TEAMS) - 1)
    probs = {team: (p if team == top else rest) for team in TEAMS}
    return Decision(provider="test", team_probs=probs, urgency=urgency, refund_prob=0.1, latency_ms=1.0)


def test_confident_answer_is_auto_handled():
    r = route(make_decision("billing", 0.96), threshold=0.90)
    assert r.action == AUTO_HANDLE
    assert r.team == "billing"
    assert r.reason == "Billing at 96%, above the 90% threshold"


def test_unsure_answer_goes_to_human():
    r = route(make_decision("technical", 0.70), threshold=0.90)
    assert r.action == SEND_TO_HUMAN
    assert "below the 90% threshold" in r.reason


def test_exactly_at_threshold_is_auto_handled():
    assert route(make_decision("sales", 0.90), threshold=0.90).action == AUTO_HANDLE


def test_threshold_is_a_dial():
    # Same probabilities, different threshold, different action: no model call needed.
    d = make_decision("account", 0.85)
    assert route(d, threshold=0.80).action == AUTO_HANDLE
    assert route(d, threshold=0.90).action == SEND_TO_HUMAN


def test_other_always_goes_to_human_even_when_confident():
    r = route(make_decision("other", 0.99), threshold=0.50)
    assert r.action == SEND_TO_HUMAN


def test_urgent_flag_from_urgency_4():
    assert route(make_decision("billing", 0.95, urgency=4.0), 0.9).urgent
    assert not route(make_decision("billing", 0.95, urgency=3.9), 0.9).urgent


def test_unavailable_provider_goes_to_human():
    r = unavailable("jev")
    assert r.action == SEND_TO_HUMAN
    assert r.confidence == 0.0


def test_just_below_threshold_never_displays_as_reaching_it():
    r = route(make_decision("technical", 0.8996), threshold=0.90)
    assert r.action == SEND_TO_HUMAN
    assert r.reason == "Technical at 89%, below the 90% threshold"
