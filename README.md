# Support Coach — local stack

A demo of real-time support coaching: analyze a customer message, look up the
order + policy, and draft a grounded reply the agent can edit before sending.

Three pieces:

- **`coach.py`** — the coaching logic: the model calls, the prompts, and the stub
  fact layer. Shared by the backend and the notebook, so it exists in one place only.
- **`app.py`** — FastAPI backend. HTTP wrapper around `coach.py`.
- **`index.html`** — the UI. Calls the backend; falls back to a built-in mock
  if the backend is offline (so a demo never shows a blank screen).

There's also **`support_coach.ipynb`** — the same logic as a runnable notebook,
if you'd rather present in Colab than run a server.

## What's real and what's stubbed (read this before demoing)

- **Real:** the model analysis, scoring, and reply drafting (when the backend is
  running with a key). The reply is *grounded* — it only states facts pulled from
  the order record and policy, and says "I'll check" when a fact is missing.
- **Stubbed:** the order records (`ORDERS`), policies (`POLICIES`), and the FAQ
  match cards (`FAQ_LIBRARY`). These are hand-written dicts. Swap `get_order`,
  `get_policy`, `get_faqs` for real data sources later — nothing else changes.
- **Not built:** a real FAQ retriever. The match cards are a visual stand-in.
  If a reviewer asks "where's the matcher," the honest answer is: it's mocked;
  the grounding actually comes from keyed policy lookup, not semantic search.

## Run

1. Install deps:
   ```
   pip install -r requirements.txt
   ```

2. Set your key:
   ```
   export GROQ_API_KEY=gsk_...            # Windows: set GROQ_API_KEY=...
   ```

3. Start the backend:
   ```
   uvicorn app:app --reload --port 8000
   ```

4. Open `index.html` in a browser (double-click, or serve it):
   ```
   python -m http.server 5500
   # then visit http://localhost:5500/index.html
   ```

The status dot, top right: **green** = talking to the live backend with a key set,
**amber** = running on the built-in mock, with the reason in the label
(`backend offline`, `no GROQ_API_KEY`, or `backend error`). A backend that is
up but keyless still reads amber — it cannot do real analysis, so the dot never
shows green over mock output.

## Endpoints

- `GET  /health` — `{status, key_set}`
- `GET  /queue` — the open backlog, priority-ordered → `{tickets[], count}`
- `POST /analyze` — `{message, history[]}` → analysis + facts + suggested reply
- `POST /evaluate` — `{customer_message, agent_message}` → scores + coaching tip

## The ticket queue

The **Queue** tab is the landing screen: every open ticket, ordered by how much
it needs an agent right now rather than by arrival time.

`GET /queue` analyzes each ticket (one model call per ticket, run concurrently),
scores it, and pushes it into a real binary heap (`TicketQueue` in `coach.py`,
built on `heapq`). Draining the heap gives the work order.

Score out of 100: **risk 40 · urgency 30 · sentiment 15 · tier 10 · wait 5**.
Risk, urgency and sentiment come from analyzing the message; tier and waiting
time come from the ticket record. Waiting time saturates at 48h so age can never
out-weigh real risk — which is why the oldest ticket in the demo backlog
("no rush, just planning ahead", waiting 2 days) sorts *last*, and a laptop that
died 8 minutes ago sorts near the top. That inversion is the point of the screen.

Ties break on longest wait, then insertion order, so the list stays stable
between refreshes while an agent works down it.

"Coach this" on any row loads that ticket into the coaching workspace — that
fires `/analyze`, the normal per-message path.

## Notes

- Model defaults to `openai/gpt-oss-120b` on Groq. Check `GET /openai/v1/models` before changing it —
  Groq retires model IDs faster than its docs page updates.
  Override with `GROQ_MODEL`.
- CORS is open (`*`) for local convenience. Lock it down before any real deployment.
- The backend never handles your key beyond reading the env var; the UI never sees it.
