# Support Coach

An AI-assisted workspace for customer-support agents. It decides **who to help next**, and helps write **what to say** — without letting the model invent facts.

---

## Table of contents

- [The problem](#the-problem)
- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Implementation](#implementation)
  - [1. The coaching pipeline](#1-the-coaching-pipeline)
  - [2. Priority queue](#2-priority-queue)
  - [3. Grounded reply drafting](#3-grounded-reply-drafting)
  - [4. FAQ retrieval (BM25)](#4-faq-retrieval-bm25)
  - [5. Conversation memory](#5-conversation-memory)
  - [6. Authentication](#6-authentication)
  - [7. Agent analytics](#7-agent-analytics)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [What is real and what is stubbed](#what-is-real-and-what-is-stubbed)
- [Security notes](#security-notes)
- [What I learned](#what-i-learned)
- [Limitations and future work](#limitations-and-future-work)

---

## The problem

A support agent opening their queue faces two questions every few minutes:

1. **Which ticket do I pick up next?** Most helpdesks answer this with first-in-first-out. That is the wrong answer surprisingly often — a customer whose laptop died eight minutes ago and who cannot work is a bigger problem than a polite question that has been sitting for two days.

2. **What do I write?** Drafting is slow, and the tempting fix — an AI that writes the reply — is dangerous. A model that invents a refund window creates a promise the company has to honour.

Support Coach addresses both, and treats the second one as a safety problem rather than a writing problem.

## What it does

When a customer message arrives, the system:

1. **Analyses it** — sentiment, urgency, escalation risk, a one-line summary, and the lookup keys (order ID, issue type).
2. **Looks up the facts** — the order record and policy by exact key, plus the most relevant knowledge-base articles by ranked search.
3. **Drafts a reply** that may only state facts found in step 2. If a fact is missing, the draft says "I'll check" instead of guessing.

Every open ticket is scored from its analysis and ordered by a priority queue, so the agent works down a list ranked by need rather than by arrival time. When the agent sends a reply, it is scored for tone, empathy and clarity, and the ticket leaves the queue.

The app opens in one of two modes:

| Mode | Sign-in | Purpose |
|---|---|---|
| **Demo mode** | None | The full workspace, open to anyone. Carries no real customer data. |
| **Agent interface** | Required | What an agent actually uses. Adds a personal performance dashboard. |

## Tech stack

**Backend**

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.12 | — |
| Web framework | FastAPI 0.141 | Async, dependency injection for the auth gate, automatic OpenAPI docs |
| Server | Uvicorn 0.52 | ASGI server with hot reload |
| Validation | Pydantic 2.13 | Request/response schemas |
| LLM provider | Groq (`groq` 1.7 SDK) | Chat-completions API; very high tokens/sec, which matters for a per-message loop |
| Model | `openai/gpt-oss-120b` | Open-weight, follows the JSON contracts, fast enough to analyse a whole queue concurrently |

**Frontend**

Vanilla HTML, CSS and JavaScript in a single file. No React, no bundler, no build step — a deliberate choice for a project this size, since a build pipeline is real ongoing cost for no gain here. Charts are hand-authored inline SVG with no charting library.

**Standard library, doing real work**

The interesting algorithms use no third-party packages:

| Module | Used for |
|---|---|
| `heapq` | The priority queue — a genuine binary heap, not a sorted list |
| `hashlib`, `hmac`, `secrets` | PBKDF2 password hashing and HMAC-signed session tokens |
| `threading` | Lock around shared state, since the queue analyses tickets in parallel |
| `concurrent.futures` | Thread pool that analyses every open ticket at once |
| `math`, `collections`, `re` | BM25 scoring, tokenisation, metric aggregation |

**Storage** — JSON files on disk. No database, by design: the project is a demo, and a schema would be scope that earns nothing.

## Architecture

```
                Browser (index.html, vanilla JS)
                 │  demo mode  │  agent interface
                 └──────┬──────┘
                        │  Bearer token on every request
                 ┌──────▼──────┐
                 │   app.py    │  FastAPI: routing, CORS, auth gate
                 └──────┬──────┘
        ┌───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼              ▼
   coach.py       retrieval.py       store.py       auth.py
   AICoach        BM25 index         history        PBKDF2 +
   prompts        over faqs.json     ticket state   HMAC tokens
   TicketQueue    (209 articles)     events            │
   stub facts                           │              │
        │                               ▼              │
        │                          analytics.py        │
        ▼                          metrics             │
   Groq API                                            │
   (gpt-oss-120b)                                 agents.json
```

`app.py` is routing only. All logic lives in the modules beneath it, so the same code runs from the FastAPI backend and from the Jupyter notebook without duplication.

## Implementation

### 1. The coaching pipeline

Two model calls per customer turn, in a fixed order — and the order is a **data dependency, not a style choice**:

```
customer message
      │
      ▼
  [call 1] analyse ──► sentiment, urgency, escalation risk,
      │                order_id, issue_type
      ▼
  keyed lookups: get_order(order_id), get_policy(issue_type)
  ranked search: BM25 over the FAQ corpus
      │
      ▼
  [call 2] draft reply, constrained to the facts just gathered
```

The reply cannot be drafted until analysis has extracted the order ID and that record has been fetched — which is why these are two calls rather than one.

Scoring the agent's reply is a **separate endpoint** (`/evaluate`) because it fires on a different event: the agent's turn, not the customer's.

### 2. Priority queue

`TicketQueue` in `coach.py` is a binary heap over `heapq`, keyed `(-score, -waiting_minutes, sequence)`:

- negated because `heapq` is a min-heap and we want highest-priority first
- ties break on **longest wait**, then insertion order, so the list stays stable while an agent works down it

The score is out of 100:

| Factor | Points | Source |
|---|---|---|
| Escalation risk | 40 | model analysis |
| Urgency | 30 | model analysis |
| Sentiment | 15 | model analysis |
| Customer tier | 10 | ticket record |
| Waiting time | 5 | ticket record |

**Waiting time saturates at 48 hours.** This is the design decision that makes the screen worth having: past two days, age stops accumulating, so it can never out-weigh genuine risk. In the demo backlog the oldest ticket ("no rush, just planning ahead", waiting two days) sorts *last*, while a laptop that died eight minutes ago sorts near the top.

Every open ticket is analysed concurrently on a thread pool — a serial loop would make the dashboard feel dead. Resolved tickets are filtered out *before* analysis, so a handled ticket never costs a model call again.

### 3. Grounded reply drafting

This is the safety property of the project.

The drafting prompt injects a `KNOWN FACTS` block — the order record, the policy, and the retrieved articles — and forbids stating anything outside it:

```
KNOWN FACTS (the only facts you may state as company policy or order status):
--- Order record ---            {...}
--- Relevant policy ---         {...}
--- Retrieved knowledge-base articles ---   FAQ-1002: ...

Rules:
- You MAY state specific numbers/status ONLY if they appear in KNOWN FACTS above.
- This includes procedural detail: do not invent durations, step counts,
  button timings, or instructions that are not written in KNOWN FACTS.
- If a fact the customer wants is NOT in KNOWN FACTS, say you will check
  rather than inventing it.
```

In practice this is why a draft says *"your $1,499 refund has been pending for 20 days, which exceeds our 5-7 business-day window"* — every one of those numbers came from a looked-up record — and why an off-topic question produces "let me look into that" instead of a confident fabrication.

### 4. FAQ retrieval (BM25)

`retrieval.py` implements **Okapi BM25**, the ranking function behind Lucene and Elasticsearch, over a corpus of **209 knowledge-base articles** in `faqs.json` spanning 12 categories.

- IDF weighting and document-length normalisation (`k1=1.5`, `b=0.75`)
- **Field weighting** — a term in the title counts 3×, in tags 2×, in the body 1×
- **Query expansion** — a synonym map for support vocabulary ("money back" → refund). Expansion terms score at **0.45** of a word the customer actually typed, so a generic expansion cannot outrank a document that matched the real wording
- **Order IDs are stripped before ranking.** `WORD-88221` tokenises to `word` + `88221` and drags in whichever article documents the ID format. They are lookup keys, handled exactly by `get_order`, and pure noise as search terms
- The analyser's `issue_type` is a **boost, not a filter** — the whole corpus is always searched

**Two gates** decide whether a result is shown, because neither works alone: an absolute score floor, and a floor that scales with query length but is **capped at five content words**. BM25 score does not grow with query length — only *matching* terms contribute — so an uncapped per-term floor punishes detailed messages. Off-topic input retrieves nothing and the UI says "no strong match, escalate" rather than showing the best of a bad set.

Scores displayed in the UI are the **real BM25 values**, not percentages. BM25 is unbounded, so a "%" would invent precision it does not have.

### 5. Conversation memory

`store.py` keeps a per-ticket transcript and open/resolved state in a JSON file behind a threading lock.

The lifecycle:

- a customer message appends to the thread and opens the ticket
- the agent's reply is recorded and the ticket **resolves**, removing it from the queue
- if the customer writes again, `/analyze` **reopens** it — resolving is never a one-way door

Both the analyser and the drafter receive the transcript, and it changes the result substantially. The same follow-up message, analysed with and without history:

| *"How much longer do I have to wait?"* | No history | With history |
|---|---|---|
| Sentiment | neutral | **negative** |
| Escalation risk | **low** | **high** |
| Key issue | "waiting for response" | "refund not received" |
| Order found | ✗ | ✓ **WORD-88221** |

The order is found because the ID appears earlier in the *thread*, not in the message being analysed. Without history that ticket scores roughly 40 points lower and sinks in the queue — which is exactly the customer about to churn.

### 6. Authentication

Sign-in is **enforced by the API, not by the UI**. `/queue`, `/analyze`, `/evaluate` and `/tickets/*` all return `401` without a valid token, so customer queries are not protected by hiding a panel.

- **PBKDF2-SHA256**, 200,000 iterations, per-password salt, constant-time comparison
- **HMAC-signed session tokens** carrying subject, role and expiry, verified on every request
- Unknown-email and wrong-password return **identical** error text, so the endpoint cannot be used to discover which addresses exist
- Demo mode also carries a token (from `/auth/demo`), so there is one auth path through the API rather than an unauthenticated bypass branch. Its `demo` role cannot read agent analytics — that returns `403`

See [Security notes](#security-notes) for what this deliberately is not.

### 7. Agent analytics

The **Agent dashboard** tab is computed entirely from real activity. There is no seeded history: a new agent sees an empty state explaining what will appear, and the charts fill as tickets are handled.

The headline metric is **measured, not guessed**. When an agent replies, the system stores how the customer sounded beforehand; when that customer writes back, it stores how they sound then. Comparing the two says whether the reply calmed them down or made things worse.

Because only tickets where the customer replied can be measured, the panel always reports `measured` alongside `handled` — quoting the split alone would overstate the sample.

Also shown: tickets per day over 14 days, tone/empathy/clarity as three small multiples, issue and risk mix, and the agent's most repeated coaching tips.

Charts are hand-authored inline SVG. Colours come from the app's existing tokens and were **validated for colour-vision separation and contrast rather than eyeballed**; every segment is directly labelled so identity never rests on colour alone. Days with no tickets are drawn as **gaps, not zeros** — plotting them as zero would invent a bad score.

## Project structure

```
support-coach/
├── app.py                FastAPI: routing, CORS, auth dependency        (369 lines)
├── coach.py              AICoach, prompts, stub facts, TicketQueue      (503 lines)
├── retrieval.py          Okapi BM25 index over faqs.json                (273 lines)
├── store.py              Conversation history, ticket state, events     (170 lines)
├── auth.py               PBKDF2 hashing, HMAC session tokens            (171 lines)
├── analytics.py          Agent performance metrics                      (102 lines)
├── index.html            Single-file UI: landing, login, queue,
│                         coaching workspace, dashboard, BM25 port
├── faqs.json             209-article knowledge base, 12 categories
├── support_coach.ipynb   The same logic as a runnable notebook (Colab)
├── requirements.txt
└── README.md
```

Generated at runtime and gitignored: `agents.json` (hashed credentials), `conversations.json` (transcripts, ticket state, analytics events).

## Getting started

**Prerequisites** — Python 3.10+ and a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys).

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set your API key**

```bash
setx GROQ_API_KEY "gsk_your_key_here"          # Windows — then open a new terminal
export GROQ_API_KEY=gsk_your_key_here          # macOS / Linux
```

Optionally set `SUPPORT_COACH_SECRET` to keep sessions valid across restarts. Without it the token signing key is regenerated per process and everyone is signed out on restart.

**3. Start the backend** (from the project folder)

```bash
python -m uvicorn app:app --reload --port 8000
```

**4. Serve the UI** in a second terminal

```bash
python -m http.server 5500
```

**5. Open** [http://localhost:5500/index.html](http://localhost:5500/index.html)

Demo accounts are seeded into `agents.json` on first run and printed to the console:

```
agent@support.local  /  demo1234
lead@support.local   /  demo1234
```

**Status indicator** — the dot in the header reads **green** when talking to a live backend with a key set, and **amber** when running on the built-in offline mock, with the reason shown (`backend offline`, `no GROQ_API_KEY`, `backend error`). A backend that is running but keyless still reads amber, because it cannot do real analysis; the dot never shows green over mock output.

**Model override** — defaults to `openai/gpt-oss-120b`. Set `GROQ_MODEL` to change it, but check `GET https://api.groq.com/openai/v1/models` first: Groq retires model IDs faster than its documentation is updated.

## API reference

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Status and whether an API key is set |
| `POST` | `/auth/login` | none | `{email, password}` → `{token, agent}` |
| `POST` | `/auth/demo` | none | A demo-role token, no credentials required |
| `GET` | `/auth/me` | token | The current session's agent |
| `GET` | `/queue` | token | Open backlog, priority-ordered |
| `POST` | `/analyze` | token | Analysis + facts + grounded draft reply |
| `POST` | `/evaluate` | token | Scores + coaching tip; records the turn, resolves the ticket |
| `GET` | `/analytics` | **agent** | That agent's performance metrics |
| `GET` | `/tickets/{id}` | token | Stored thread and open/resolved state |
| `POST` | `/tickets/{id}/reopen` | token | Return a handled ticket to the queue |
| `POST` | `/tickets/reset` | token | Clear all history (demo reset) |

Interactive docs at `http://localhost:8000/docs`.

## What is real and what is stubbed

Being precise about this line is a design goal, not an afterthought — a demo that blurs it teaches the wrong thing about the system.

**Real**

- Model analysis, reply drafting and reply scoring
- The priority ordering — a genuine binary heap over scores derived from real analysis
- FAQ retrieval — real BM25 with real scores over a real 209-article corpus
- Authentication — real password hashing and a real, server-enforced boundary
- Agent analytics — computed from actual handled tickets, with no seeded history

**Stubbed**

`ORDERS`, `POLICIES` and `TICKETS` are hand-written dictionaries in `coach.py`. `get_order`, `get_policy` and `get_tickets` are the swap points: replace them with a database or helpdesk API and nothing downstream changes.

**Honest limitation**

BM25 is *lexical*. It matches words, not meaning. A question that shares no vocabulary with any article will not retrieve it, however related it is. The synonym map closes the common gaps; it is not a substitute for embeddings.

## Security notes

The authentication is **demo-grade and deliberately incomplete**. It is real where it matters — hashing and server-side enforcement — but:

- **No TLS.** Over plain HTTP a password is readable on the wire.
- **No rate limiting or lockout**, so it is brute-forceable.
- **No signup, password reset, or rotation.** Accounts are provisioned, which suits company-to-company use, but there is no recovery path.
- **CORS is wide open** (`allow_origins=["*"]`) for local convenience. Lock it to a real origin before any deployment.
- Seed passwords appear in `auth.py` as documented demo credentials.

Do not put this on a public network as-is.

## What I learned

**Prompts that ask for a rating need a rubric, or everything becomes "high".** My first queue put five of six tickets at high escalation risk — technically responsive, completely useless for ranking. The model had no definition of the levels, so it invented one. Adding explicit criteria plus one calibration line ("in a normal queue most tickets are low or medium; if you mark everything high, the ranking is worthless") spread the scores from 98/95/95/89/72/25 to 84/77/77/72/57/25, and made the ordering mean something.

**Reasoning models spend their output budget before producing output.** After lengthening a prompt I started getting `400 json_validate_failed` with an empty `failed_generation`. The model was consuming all of `max_completion_tokens` on internal reasoning and emitting nothing. Lowering `reasoning_effort` and raising the budget fixed it. The error message pointed at JSON validation, which is three steps from the actual cause.

**Provider documentation goes stale faster than the API does.** I built against `llama-3.3-70b-versatile` because Groq's own model page listed it; the live API returned `model_not_found`. Querying `GET /openai/v1/models` with the actual key is the only reliable source. I now check the live list before pinning any model ID.

**A threshold that scales with input length can punish the very input it should reward.** My retrieval gate scaled with query length to reject rambling off-topic messages. But BM25 score does not grow with query length — only matching terms contribute. A customer who wrote "...I need this sorted before my trip on Friday" added seven words to the floor and nothing to the score, so the *more* detail they gave, the harder it became to help them. Capping the scale factor fixed it; sweeping the cap against a labelled set of on-topic and off-topic queries proved which value to use.

**Grounding rules have to cover procedure, not just numbers.** My original rule blocked inventing figures like refund windows. Then a draft said "hold the power button for about 10 seconds" — a fabricated procedural detail, from an article that only said "force restart". Low harm, but the same failure mode. Adding retrieval widened the surface for invention, because articles describe *steps*, not just values.

**A login that only hides the UI is not security.** The obvious implementation gates the frontend. But `/queue` returns customer data to anyone who calls it directly, so the boundary has to live in the API. Building it as a FastAPI dependency made it hard to forget: a new endpoint that returns customer data must declare the dependency to compile into the route.

**Conversation history is not a nice-to-have for analysis, it changes the answer.** I expected history to improve the drafted wording. It also moved a follow-up from *low* risk to *high*, and let the system find an order record whose ID appeared only in an earlier turn. Measuring the same message with and without context was more convincing than any amount of reasoning about it.

**Refusing to fake data makes the product more honest and the design harder.** Choosing live-only analytics meant the dashboard opens empty, so the empty state had to explain what will appear and why. Similarly, days with no tickets are gaps rather than zeros, and the sentiment metric always reports how many tickets it could actually measure. Each of those is a small extra effort that prevents a confidently wrong impression.

**Colour accessibility is computable, so it should be computed.** My instinct for three chart series — blue, violet, teal — was ΔE 1.7 apart under deuteranopia: effectively identical. Rather than hunt for three safe hues, the better answer was to remove categorical colour entirely and use three small multiples, each a single series. Validating beat guessing, and it changed the design rather than just the palette.

**Duplicated logic is the bug you ship.** The project started with the coach class, prompts and stub data copied into both the backend and the notebook. Editing one and forgetting the other was the most likely failure in the codebase. Extracting a shared module before any feature work was the right first move, and everything since has been a single-source change.

## Limitations and future work

- **Data sources are stubbed.** Replacing `get_order` / `get_policy` / `get_tickets` with a real database or helpdesk API is the main step from demo to usable, and it is a data-access task rather than a modelling one.
- **Retrieval is lexical.** Embeddings would handle paraphrase that shares no vocabulary. Groq offers no embeddings endpoint, so this needs a local model or a second provider.
- **No automated tests.** The highest-value first test asserts the grounding rule: given an order with no `refund_amount`, the drafted reply must not state one.
- **Analytics are per-agent only.** Team-level views and comparison across agents are not built.
- **Storage is a JSON file.** Fine for one process; concurrent deployments would need a real database.
