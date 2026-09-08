# Support Coach — local stack

A demo of real-time support coaching: analyze a customer message, look up the
order + policy, and draft a grounded reply the agent can edit before sending.

Four pieces:

- **`coach.py`** — the coaching logic: the model calls, the prompts, and the stub
  fact layer. Shared by the backend and the notebook, so it exists in one place only.
- **`retrieval.py`** — BM25 retrieval over `faqs.json` (209 articles).
- **`store.py`** — conversation history and open/resolved ticket state.
- **`app.py`** — FastAPI backend. HTTP wrapper around `coach.py`.
- **`index.html`** — the UI. Calls the backend; falls back to a built-in mock
  if the backend is offline (so a demo never shows a blank screen).

There's also **`support_coach.ipynb`** — the same logic as a runnable notebook,
if you'd rather present in Colab than run a server.

## What's real and what's stubbed (read this before demoing)

- **Real:** the model analysis, scoring, and reply drafting (when the backend is
  running with a key). The reply is *grounded* — it only states facts pulled from
  the order record and policy, and says "I'll check" when a fact is missing.
- **Real:** FAQ retrieval. `retrieval.py` runs Okapi BM25 — the ranking function
  behind Lucene — over the 209-article corpus in `faqs.json`. The scores on the
  cards are real BM25 scores, not percentages, and an off-topic message retrieves
  nothing rather than showing the best of a bad set.
- **Stubbed:** the order records (`ORDERS`), policies (`POLICIES`) and the ticket
  backlog (`TICKETS`), all hand-written dicts in `coach.py`. Swap `get_order`,
  `get_policy`, `get_tickets` for real data sources later — nothing else changes.
- **Honest limitation:** BM25 is *lexical*. It matches words, not meaning. A
  question sharing no vocabulary with any article will not retrieve it. A small
  query-side synonym map closes the common gaps ("money back" -> refund); it is
  not a substitute for embeddings. Say this plainly if asked.

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

## FAQ retrieval

`retrieval.py` builds an in-memory BM25 index over `faqs.json` at import:
IDF weighting, length normalisation, and field weighting (a term in the title
counts 3x a term in the body). Two gates decide whether a hit is shown — an
absolute score floor and a per-query-term floor — so a long off-topic message
cannot accumulate weak partial hits into a false match.

The analyzer's `issue_type` is a **boost**, not a filter: the whole corpus is
always searched, so a billing article can still win on a ticket classified as a
refund. Order IDs are stripped from the query before ranking; they are lookup
keys handled exactly by `get_order`, and as search terms they are pure noise.

Retrieved articles are injected into the reply prompt's KNOWN FACTS block, so
the draft can cite real article content under the same rule as everything else:
state nothing that is not in the block.

`index.html` carries a JavaScript port of the same scoring, reading the same
`faqs.json`, so the offline mock ranks the way the backend ranks. **The
constants are mirrored in both files — change them together.**

## Endpoints

- `GET  /health` — `{status, key_set}`
- `GET  /queue` — open backlog, priority-ordered → `{tickets[], count, resolved_count}`
- `POST /analyze` — `{message, ticket_id?, history[]}` → analysis + facts + reply
- `POST /evaluate` — `{customer_message, agent_message, ticket_id?, resolve?}` → scores + tip
- `GET  /tickets/{id}` — stored thread + open/resolved
- `POST /tickets/{id}/reopen` — put a handled ticket back in the queue
- `POST /tickets/reset` — clear all history (demo reset)

## Conversation history and ticket lifecycle

`store.py` keeps a per-ticket transcript and an open/resolved flag in
`conversations.json` — a JSON file behind a lock, not a database. It is
gitignored: runtime state, not source.

**The lifecycle.** A customer message is appended to the thread and the ticket
is (re)opened. When the agent sends a reply, that turn is recorded and the
ticket resolves, which removes it from the queue — a handled ticket is not
re-analysed on every refresh. If the customer writes in again, `/analyze`
reopens it automatically, so resolving is never a one-way door.

**History is context, and it changes the answer.** Both the analyzer and the
drafter receive the transcript. The same follow-up, analysed with and without
it:

| *"How much longer do I have to wait?"* | No history | With history |
|---|---|---|
| sentiment | neutral | negative |
| escalation risk | **low** | **high** |
| key issue | "waiting for response" | "refund not received" |
| order found | no | **yes — WORD-88221** |

The order is found because the ID appears earlier in the thread, not in the
message being analysed. Without history that message scores ~40 points lower and
sinks in the queue, which is exactly the customer who is about to churn.

Pass no `ticket_id` and the endpoints stay stateless, as they were.

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
