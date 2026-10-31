# rlcd-ticket-router

A small localhost demo that routes support tickets with two decision models,
**Jev** (closed, by TypeSafe AI, called over its API) and **Laya** (open weights,
runs on your own GPU or CPU), and shows why their **probabilities** are useful.

Paste a ticket, pick a model, click **Route it**. Each model answers three typed questions:

| Question | Type | Answer |
|---|---|---|
| `team` | choice | billing, technical, sales, account, other (+ a probability for each) |
| `urgency` | score | 1 to 5 |
| `refund_requested` | yes/no | probability of "yes" |

Then one simple rule (in `app/router.py`) turns that into an action:

- **AUTO-HANDLE** if the top team's probability is at or above the threshold and the team is not `other`
- otherwise **SEND TO HUMAN**
- urgency 4 or more gets an **URGENT** flag

The threshold slider re-applies the rule in the browser **without calling the model again**.
The probability is a dial you set to your own risk level.

> **Note:** any results shown in this demo or the video come from my own tests on my own
> machine. They are not claims made by the model authors.

## Quick start (mock mode, no key, no download)

Needs Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and choose **Mock (no model)**. The mock is a keyword counter,
so it's deterministic and good for trying out the page. It is not a real model.

Run the tests with `pytest`.

## Laya mode (local, open weights)

Install PyTorch first, picking the build that fits your machine, then Laya:

```bash
# NVIDIA GPU: pick the CUDA build your driver supports (nvidia-smi shows "CUDA Version").
# I used CUDA 13.2 for an RTX 5060:
pip install torch --index-url https://download.pytorch.org/whl/cu132
# ...or no NVIDIA GPU: the smaller CPU-only build
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install laya
python scripts/probe_laya.py      # downloads the model once, prints raw output per question type
```

- Model: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (English checkpoint,
  ModernBERT-large, about 421M parameters, Apache-2.0). Code: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya).
- The weights (~800 MB) are cached in `.model-cache/` in this folder (git-ignored), so they download only once.
- Laya runs on the GPU when PyTorch can see one and on the CPU otherwise. Set `LAYA_DEVICE=cpu`
  in `.env` to force CPU, for example to show the speed difference. The card shows where it ran.
- **First load:** the first run downloads the model, which took about 90 s on my connection.
  After that, loading from the cache takes about 2 to 3 s. The app loads Laya on the first Laya
  request (plus one warm-up call, so the GPU's one-off setup doesn't show up as a slow ticket),
  so that first click is slower. Later clicks reuse the loaded model.
- My timings for one ticket (all three questions in one forward pass), same 6 example tickets:
  about **20 ms on an RTX 5060** and about **0.4 to 0.55 s on a Ryzen 5 9600X CPU** (12 threads).
  Decisions were the same on both; probabilities differed by at most 0.01.

Then start the app and choose **Laya**.

## Jev mode (hosted, needs an API key)

1. Create a key in the TypeSafe console (https://console.typesafe.ai/keys) and put it in `.env`:
   `TYPESAFE_API_KEY=...`
2. Check the raw responses: `python scripts/probe_jev.py`
3. Start the app and choose **Jev**, or **Compare both** to see Jev and Laya side by side.

Requests go straight to TypeSafe: `POST https://api.typesafe.ai/v1/systemone` with model
`jev-latest` ([API reference](https://docs.typesafe.ai/api/system-one.md)). TypeSafe bills
input tokens only, and each Jev card shows that ticket's input tokens and cost:
tokens reported by Jev x the listed price of $0.042 per million input tokens
([TypeSafe models page](https://docs.typesafe.ai/models.md), checked 2026-10-30; set in
`app/providers/jev.py`). In my tests one ticket with these three questions used about
520 to 560 input tokens, so about $0.000023 per ticket. Your TypeSafe console has the real bill.
The cost is also shown in rupees, using a live USD to INR rate (European Central Bank
reference rate via [frankfurter.dev](https://frankfurter.dev), no key needed, refreshed
every 6 hours). The page shows the rate and its date. If the rate can't be fetched, the
app uses `USD_INR_RATE` from `.env`, or shows USD only.
The key is read from `.env` and is never printed or logged.

## If something fails

A missing key, a timeout, a model that isn't downloaded or an unexpected reply shows a message
on that model's card and routes the ticket to **SEND TO HUMAN**. In Compare mode the other
model's card still shows its result.

## Layout

```
app/
  main.py              FastAPI: GET / (page), POST /route, GET /config
  router.py            the threshold rule
  providers/
    base.py            Provider interface, Decision, the three shared questions
    jev.py             Jev over HTTP
    laya.py            Laya locally (GPU or CPU)
    mock.py            keyword mock for offline use and tests
  static/index.html    the page (plain HTML + JS)
scripts/
  probe_jev.py         one real Jev request per question type, raw JSON
  probe_laya.py        one Laya run per question type, raw output
tests/                 pytest: routing rule, mock, API error handling, parsers
```

`POST /route` takes `{"ticket": "...", "provider": "jev" | "laya" | "mock" | "both", "threshold": 0.9}`
(`threshold` is optional and defaults to `DEFAULT_THRESHOLD` from `.env`).

## Things to know

- Both models get the exact same questions and option descriptions (in `app/providers/base.py`),
  so differences come from the models, not from the prompts.
- "Confidence" here means the probability of the top team. Laya's output also has its own
  `confidence` field, which is a different number; this demo doesn't use it.
- Laya's model card says its probabilities ship over-confident and recommends re-calibrating
  on your own data before you trust them. Laya also prints a warning on load that some of its
  calibration temperatures are invalid. This demo uses the probabilities as they come.
