"""Laya (open weights, Apache-2.0), running locally on your GPU, or CPU if there is none.

Real output from scripts/probe_laya.py (trimmed):
  {"answers": {
      "team":             {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.96, ...}},
      "urgency":          {"type": "score", "score": 1.26, "legend": {"0": "1: no rush", ...},
                           "probabilities": {"0": 0.26, ..., "4": 0.006}},
      "refund_requested": {"type": "noul", "noul": 0.85}},
   "routing": {"model": "english", "repo": "convaiinnovations/laya", ...}}
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

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

# Keep the downloaded weights inside the repo (git-ignored). Must be set before
# laya / huggingface_hub is imported, which is why laya is imported lazily below.
os.environ.setdefault("HF_HUB_CACHE", str(Path(__file__).resolve().parents[2] / ".model-cache"))

log = logging.getLogger("uvicorn.error")

# "english" = convaiinnovations/laya (ModernBERT-large). Pinned so Laya's
# language auto-routing never swaps in the multilingual checkpoint mid-demo.
CHECKPOINT = "english"

QUESTIONS: dict[str, dict] = {
    "team": {"type": "choice", "instructions": TEAM_INSTRUCTIONS, "criteria": TEAMS},
    "urgency": {"type": "score", "instructions": URGENCY_INSTRUCTIONS, "criteria": URGENCY_LEVELS},
    "refund_requested": {"type": "noul", "instructions": REFUND_INSTRUCTIONS},  # noul = Laya's yes/no
}


def pick_device() -> str:
    """GPU if there is one. LAYA_DEVICE=cpu in .env forces CPU, e.g. to compare speeds."""
    wanted = os.getenv("LAYA_DEVICE", "auto").strip().lower()
    if wanted in ("cpu", "cuda"):
        return wanted
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class LayaProvider(Provider):
    name = "laya"

    def __init__(self) -> None:
        self._router: Any = None
        self._lock = threading.Lock()  # Compare mode can ask twice at once; load only once

    def warm_up(self) -> None:
        with self._lock:
            if self._router is not None:
                return
            try:
                from laya import Router
            except ImportError:
                raise ProviderError("Laya is not installed. Run: pip install laya")
            device = pick_device()
            log.info("Loading Laya '%s' checkpoint on %s (slow the first time)...", CHECKPOINT, device)
            start = time.perf_counter()
            try:
                router = Router(device=device)
                router.preload([CHECKPOINT])
                # One throwaway call: a GPU's first run sets up its kernels, which would
                # otherwise make the first ticket on screen look slower than the rest.
                router.predict("warm-up", QUESTIONS, model=CHECKPOINT)
            except Exception as e:
                raise ProviderError(f"Could not load the Laya model on {device} ({type(e).__name__}: {e}).")
            log.info("Laya ready on %s in %.1fs", device, time.perf_counter() - start)
            self.runs_on = "your GPU" if device == "cuda" else "your CPU"
            self._router = router

    def _answer(self, ticket_text: str) -> Answer:
        # All three questions are answered in one forward pass.
        result = self._router.predict(ticket_text, QUESTIONS, model=CHECKPOINT)
        return parse_result(result)


def parse_result(result: dict) -> Answer:
    """Turn Laya's output into an Answer (urgency on a 1-5 scale)."""
    try:
        answers = result["answers"]
        probs = answers["team"]["probabilities"]
        team_probs = {team: float(probs.get(team, 0.0)) for team in TEAMS}
        # Laya's score is the expected rung index starting at 0 (see "legend"), so +1 gives 1 to 5.
        urgency = float(answers["urgency"]["score"]) + 1
        refund_prob = float(answers["refund_requested"]["noul"])
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ProviderError(f"Unexpected Laya output format (missing {e}).")
    input_tokens = (result.get("usage") or {}).get("input_tokens")
    return Answer(team_probs, urgency, refund_prob, input_tokens)
