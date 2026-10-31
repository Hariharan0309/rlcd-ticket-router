"""Live USD -> INR rate, so Jev's cost can be shown in rupees too.

Rates come from Frankfurter (European Central Bank reference rates, published once
per working day), so we fetch at most every few hours and reuse the result.
"""

import os
import time

import httpx
from pydantic import BaseModel

RATE_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR"
CACHE_S = 6 * 60 * 60


class Rate(BaseModel):
    inr_per_usd: float
    date: str  # the day the rate was published
    source: str


_cache: tuple[float, Rate] | None = None  # (fetched at, rate)


def usd_to_inr() -> Rate | None:
    """Today's rate, else the last one we had, else USD_INR_RATE from .env, else None."""
    global _cache
    if _cache and time.monotonic() - _cache[0] < CACHE_S:
        return _cache[1]
    try:
        resp = httpx.get(RATE_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = Rate(inr_per_usd=float(data["rates"]["INR"]), date=data["date"], source="ECB via frankfurter.dev")
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        if _cache:  # offline: an older rate beats no rate
            return _cache[1]
        fallback = os.getenv("USD_INR_RATE")
        if not fallback:
            return None  # the page then shows USD only
        rate = Rate(inr_per_usd=float(fallback), date="fixed", source="USD_INR_RATE in .env")
    _cache = (time.monotonic(), rate)
    return rate
