"""FastAPI app: one page, one routing endpoint.

Run:  uvicorn app.main:app --port 8000
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.currency import usd_to_inr
from app.providers.base import Decision, Provider, ProviderError
from app.providers.jev import JevProvider
from app.providers.laya import LayaProvider
from app.providers.mock import MockProvider
from app.router import Routing, route, unavailable

load_dotenv()
log = logging.getLogger("uvicorn.error")

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.90"))
INDEX_HTML = Path(__file__).parent / "static" / "index.html"

# Created once so Laya's weights and Jev's HTTP connection are reused across requests.
PROVIDERS: dict[str, Provider] = {
    "jev": JevProvider(),
    "laya": LayaProvider(),
    "mock": MockProvider(),
}

app = FastAPI(title="rlcd-ticket-router")


class RouteRequest(BaseModel):
    ticket: str = Field(min_length=1, max_length=5000)
    provider: Literal["jev", "laya", "mock", "both"]
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class CardResult(BaseModel):
    provider: str
    decision: Decision | None  # None when the provider failed
    routing: Routing
    error: str | None = None


def run_provider(name: str, ticket: str, threshold: float) -> CardResult:
    """Ask one model. Any failure becomes a SEND TO HUMAN card, never a crash."""
    try:
        decision = PROVIDERS[name].decide(ticket)
    except ProviderError as e:
        return CardResult(provider=name, decision=None, routing=unavailable(name), error=str(e))
    except Exception as e:  # a bug or library error: log it, keep the page alive
        log.exception("%s provider crashed", name)
        error = f"{name} failed unexpectedly ({type(e).__name__}). See the server log."
        return CardResult(provider=name, decision=None, routing=unavailable(name), error=error)
    return CardResult(provider=name, decision=decision, routing=route(decision, threshold))


@app.post("/route")
async def route_ticket(req: RouteRequest) -> list[CardResult]:
    names = ["jev", "laya"] if req.provider == "both" else [req.provider]
    # Providers block (HTTP call / CPU inference), so run each in a thread.
    # In Compare mode both run at once, and each card keeps its own result or error.
    tasks = [asyncio.to_thread(run_provider, n, req.ticket, req.threshold) for n in names]
    return list(await asyncio.gather(*tasks))


@app.get("/config")
def config() -> dict:
    rate = usd_to_inr()
    return {"default_threshold": DEFAULT_THRESHOLD, "usd_inr": rate.model_dump() if rate else None}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
