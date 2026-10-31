import httpx
import pytest

from app import currency


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(currency, "_cache", None)
    monkeypatch.delenv("USD_INR_RATE", raising=False)


def fake_get(status: int, payload: dict):
    def get(url, timeout):
        return httpx.Response(status, json=payload, request=httpx.Request("GET", url))

    return get


def offline(url, timeout):
    raise httpx.ConnectError("no network")


def test_live_rate_is_used_and_cached(monkeypatch):
    monkeypatch.setattr(currency.httpx, "get", fake_get(200, {"date": "2026-10-30", "rates": {"INR": 94.95}}))
    rate = currency.usd_to_inr()
    assert rate.inr_per_usd == 94.95
    assert rate.date == "2026-10-30"
    # A second call within the cache window must not hit the network.
    monkeypatch.setattr(currency.httpx, "get", offline)
    assert currency.usd_to_inr().inr_per_usd == 94.95


def test_offline_falls_back_to_env_rate(monkeypatch):
    monkeypatch.setattr(currency.httpx, "get", offline)
    monkeypatch.setenv("USD_INR_RATE", "95")
    rate = currency.usd_to_inr()
    assert rate.inr_per_usd == 95.0
    assert rate.source == "USD_INR_RATE in .env"


def test_offline_without_fallback_shows_usd_only(monkeypatch):
    monkeypatch.setattr(currency.httpx, "get", offline)
    assert currency.usd_to_inr() is None
