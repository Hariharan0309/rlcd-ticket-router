import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.providers.base import TEAMS, ProviderError
from app.providers.jev import JevProvider, parse_response
from app.providers.laya import parse_result
from app.providers.mock import MockProvider

client = TestClient(main.app)


def test_mock_is_deterministic_and_well_formed():
    mock = MockProvider()
    a = mock.decide("I was charged twice, please refund the duplicate payment")
    b = mock.decide("I was charged twice, please refund the duplicate payment")
    assert a.team_probs == b.team_probs
    assert set(a.team_probs) == set(TEAMS)
    assert sum(a.team_probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert 1 <= a.urgency <= 5
    assert a.latency_ms >= 0
    assert a.cost_usd is None  # the mock reports no tokens, so no cost is shown


def test_mock_routes_clear_billing_ticket():
    d = MockProvider().decide("I was charged twice on my invoice, please refund the duplicate payment")
    assert d.top_team == "billing"
    assert d.confidence >= 0.9
    assert d.refund_prob > 0.5


def test_mock_sends_keywordless_ticket_to_other():
    d = MockProvider().decide("Does anyone have a good banana bread recipe?")
    assert d.top_team == "other"
    assert d.refund_prob < 0.5


def test_route_endpoint_with_mock():
    resp = client.post("/route", json={"ticket": "My login password does not work", "provider": "mock"})
    assert resp.status_code == 200
    [card] = resp.json()
    assert card["provider"] == "mock"
    assert card["decision"]["team_probs"]
    assert card["routing"]["action"] in ("AUTO-HANDLE", "SEND TO HUMAN")


def test_one_failing_provider_does_not_hide_the_other(monkeypatch):
    def broken(_ticket):
        raise ProviderError("No TYPESAFE_API_KEY in .env")

    def crashing(_ticket):
        raise RuntimeError("boom")

    monkeypatch.setattr(main.PROVIDERS["jev"], "decide", broken)
    monkeypatch.setattr(main.PROVIDERS["laya"], "decide", crashing)
    resp = client.post("/route", json={"ticket": "hello", "provider": "both"})
    assert resp.status_code == 200
    jev, laya = resp.json()
    assert jev["error"] == "No TYPESAFE_API_KEY in .env"
    assert jev["routing"]["action"] == "SEND TO HUMAN"
    assert "RuntimeError" in laya["error"]
    assert laya["routing"]["action"] == "SEND TO HUMAN"


def test_compare_mode_shows_working_provider_next_to_failed_one(monkeypatch):
    def broken(_ticket):
        raise ProviderError("timed out")

    monkeypatch.setattr(main.PROVIDERS["jev"], "decide", broken)
    monkeypatch.setattr(main.PROVIDERS["laya"], "decide", MockProvider().decide)
    jev, laya = client.post("/route", json={"ticket": "app crashes", "provider": "both"}).json()
    assert jev["decision"] is None
    assert laya["decision"] is not None


def test_laya_parser_on_real_probe_output():
    # Shape copied from scripts/probe_laya.py output.
    raw = {"answers": {
        "team": {"type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.9632, "technical": 0.0189, "other": 0.0179}},
        "urgency": {"type": "score", "score": 1.2648, "probabilities": {"0": 0.26, "1": 0.23, "2": 0.49}},
        "refund_requested": {"type": "noul", "noul": 0.8507},
    }, "usage": {"input_tokens": 55, "output_tokens": 0}}
    team_probs, urgency, refund, tokens = parse_result(raw)
    assert team_probs["billing"] == 0.9632
    assert team_probs["sales"] == 0.0  # options missing from the reply count as 0
    assert urgency == pytest.approx(2.2648)
    assert refund == 0.8507
    assert tokens == 55


def test_jev_parser_on_real_probe_output():
    # Shape copied from scripts/probe_jev.py output (model jev-1.14.0).
    raw = {"model": "jev-1.14.0", "answers": {
        "team": {"type": "choice", "choice": "billing", "confidence": 1.0,
                 "probabilities": {"billing": 1.0, "technical": 0.0, "other": 0.0}},
        "urgency": {"type": "score", "score": 2.05, "confidence": 0.8,
                    "probabilities": {"0": 0.0, "1": 0.09, "2": 0.76, "3": 0.15, "4": 0.0}},
        "refund_requested": {"type": "noul", "noul": 0.99},
    }, "usage": {"input_tokens": 346, "output_tokens": 38}}
    team_probs, urgency, refund, tokens = parse_response(raw)
    assert team_probs["billing"] == 1.0
    assert team_probs["account"] == 0.0
    assert urgency == pytest.approx(3.05)
    assert refund == 0.99
    assert tokens == 346


@pytest.mark.parametrize("parser", [parse_result, parse_response])
def test_parsers_reject_bad_output(parser):
    with pytest.raises(ProviderError):
        parser({"answers": {"team": {}}})


def fake_jev(monkeypatch, status: int, payload: dict) -> JevProvider:
    """A JevProvider whose HTTP calls hit a fake TypeSafe instead of the real one."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    jev = JevProvider()
    jev._client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(status, json=payload)))
    return jev


def test_jev_bad_key_gives_a_clear_message(monkeypatch):
    jev = fake_jev(monkeypatch, 401, {"detail": "Invalid API key"})
    with pytest.raises(ProviderError, match="HTTP 401: the API key is missing or invalid"):
        jev.decide("hello")


def test_jev_success_end_to_end(monkeypatch):
    payload = {"answers": {
        "team": {"type": "choice", "choice": "technical", "probabilities": {"technical": 0.95, "other": 0.05}},
        "urgency": {"type": "score", "score": 3.2, "probabilities": {}},
        "refund_requested": {"type": "noul", "noul": 0.1},
    }, "usage": {"input_tokens": 1000, "output_tokens": 30}}
    d = fake_jev(monkeypatch, 200, payload).decide("app crashes")
    assert d.top_team == "technical"
    assert d.urgency == pytest.approx(4.2)
    # 1,000 input tokens at $0.042 per million; output tokens are free.
    assert d.input_tokens == 1000
    assert d.cost_usd == pytest.approx(0.000042)
