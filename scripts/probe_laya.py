"""Run Laya locally once per question type and print the raw output.

Usage:  python scripts/probe_laya.py

The first run downloads the English checkpoint into .model-cache/ in this repo
(git-ignored). Later runs load it from there.
"""

import json
import os
import time
from pathlib import Path

# Must be set before laya (and huggingface_hub) is imported.
os.environ.setdefault("HF_HUB_CACHE", str(Path(__file__).resolve().parents[1] / ".model-cache"))

import torch  # noqa: E402
from laya import Router  # noqa: E402

TICKET = "I was charged twice for my March invoice. Please refund the duplicate charge."

# One example per Laya question type: "choice", "score" and "noul" (Laya's yes/no type).
PROBES: dict[str, dict] = {
    "choice": {
        "team": {
            "type": "choice",
            "instructions": "Which support team should handle this ticket?",
            "criteria": {
                "billing": "charges, invoices, payments, refunds",
                "technical": "bugs, errors, crashes",
                "other": "anything else",
            },
        }
    },
    "score": {
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["1: no rush", "2: low", "3: normal", "4: high", "5: critical"],
        }
    },
    "noul": {
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund or their money back?",
        }
    },
}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    router = Router(device=device)
    # model="english" pins the ModernBERT-large checkpoint instead of auto-routing.
    router.preload(["english"])
    print(f"Model load on {device}: {time.perf_counter() - start:.1f}s\n")

    for qtype, questions in PROBES.items():
        start = time.perf_counter()
        result = router.predict(TICKET, questions, model="english")
        ms = (time.perf_counter() - start) * 1000
        print(f"=== {qtype} ({ms:.0f} ms) ===")
        print(json.dumps(result, indent=2, default=str))
        print()


if __name__ == "__main__":
    main()
