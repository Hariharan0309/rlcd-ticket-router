"""Send one real request per question type to Jev and print the raw JSON.

Usage:  python scripts/probe_jev.py      (needs TYPESAFE_API_KEY in .env)

Request shapes follow the TypeSafe API reference
(https://docs.typesafe.ai/api/system-one.md):
  choice -> "criteria" is {option: description}
  score  -> "criteria" is a list of levels, lowest to highest (scored from 0)
  noul   -> yes/no, no criteria needed
"""

import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

URL = "https://api.typesafe.ai/v1/systemone"
TICKET = "I was charged twice for my March invoice. Please refund the duplicate charge."

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
    load_dotenv()
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY is not set. Copy .env.example to .env and add your key.")

    headers = {"Authorization": f"Bearer {key}"}  # never printed
    with httpx.Client(timeout=30) as client:
        for qtype, questions in PROBES.items():
            body = {"model": "jev-latest", "state": TICKET, "questions": questions}
            start = time.perf_counter()
            resp = client.post(URL, headers=headers, json=body)
            ms = (time.perf_counter() - start) * 1000
            print(f"=== {qtype}: HTTP {resp.status_code} ({ms:.0f} ms) ===")
            try:
                print(json.dumps(resp.json(), indent=2))
            except ValueError:
                print(resp.text)
            print()


if __name__ == "__main__":
    main()
