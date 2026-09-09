"""
Support Coach — FastAPI backend
================================

HTTP wrapper around `coach.py`. All the coaching logic, prompts, and stub data
live in that module — this file is routing, CORS, and request/response models
only.

Run:
    pip install -r requirements.txt
    export GROQ_API_KEY=gsk_...                # Windows: set GROQ_API_KEY=...
    uvicorn app:app --reload --port 8000

Then open the UI (index.html) — it calls http://localhost:8000.

Endpoints:
    POST /analyze   { "message": "...", "ticket_id": "TCK-1", "history": [...] }
        -> analysis + looked-up facts + grounded suggested reply
    POST /evaluate  { "customer_message": "...", "agent_message": "...", "ticket_id": "TCK-1" }
        -> scores + tip; records the turn and resolves the ticket
    GET  /tickets/{id}          -> stored thread + open/resolved
    POST /tickets/{id}/reopen   -> put a handled ticket back in the queue
    POST /tickets/reset         -> clear all history (demo reset)
    GET  /health

Conversation history and open/resolved state live in `store.py`.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from coach import (
    AICoach,
    TicketQueue,
    clean_key,
    get_order,
    get_policy,
    get_tickets,
    priority_score,
)
from retrieval import search_faqs
from store import get_store
from analytics import compute as compute_analytics
import auth

app = FastAPI(title="Support Coach API")

# CORS: the UI is served from a different origin (file:// or another port),
# so the browser needs the API to allow it. "*" is fine for local demo only —
# tighten this to the real origin before deploying anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily create the coach so the server can boot even without a key,
# and return a clear error on the first call instead of crashing at import.
_coach: Optional[AICoach] = None


def coach() -> AICoach:
    global _coach
    if _coach is None:
        _coach = AICoach()
    return _coach


class LoginRequest(BaseModel):
    email: str
    password: str


def current_agent(authorization: Optional[str] = Header(default=None)) -> dict:
    """Reject anything without a valid session token.

    This is the actual boundary. Hiding the queue in the UI protects nothing —
    /queue, /analyze and /evaluate all carry customer data and are only safe
    because this dependency runs first.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    payload = auth.read_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Sign in required.")
    return payload


def require_real_agent(agent: dict = Depends(current_agent)) -> dict:
    """Analytics are per-agent, so demo sessions cannot read them."""
    if agent.get("role") != "agent":
        raise HTTPException(status_code=403, detail="Agent sign-in required.")
    return agent


class HistoryItem(BaseModel):
    speaker: str
    text: str


class AnalyzeRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []
    # When present, the server's stored thread is the source of truth and the
    # turn is appended to it. Omit it for a one-off stateless analysis.
    ticket_id: Optional[str] = None


class EvaluateRequest(BaseModel):
    customer_message: str
    agent_message: str
    ticket_id: Optional[str] = None
    # Sending a reply marks the ticket handled so it leaves the queue. Set
    # false to score a draft without resolving anything.
    resolve: bool = True
    # Context from the /analyze call that produced this draft. Sent by the UI
    # so the handled-event carries what the dashboard needs without re-running
    # an analysis just to label a row.
    sentiment_before: Optional[str] = None
    issue_type: Optional[str] = None
    escalation_risk: Optional[str] = None


@app.get("/health")
def health():
    """Unauthenticated on purpose — the UI needs it to show the status dot."""
    return {"status": "ok", "key_set": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/auth/login")
def login(req: LoginRequest):
    record = auth.authenticate(req.email, req.password)
    if not record:
        # One message for both unknown email and wrong password: saying which
        # is wrong tells an attacker which addresses are real.
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return {
        "token": auth.make_token(record["email"], record["name"], role="agent"),
        "agent": {"email": record["email"], "name": record["name"], "role": "agent"},
    }


@app.post("/auth/demo")
def demo_session():
    """Demo mode is open by design — it exists to be shown to anyone.

    It still gets a token so the API has one auth path rather than an
    unauthenticated bypass branch. Role "demo" cannot read agent analytics.
    """
    return {
        "token": auth.make_demo_token(),
        "agent": {"email": "demo@local", "name": "Demo", "role": "demo"},
    }


@app.get("/auth/me")
def me(agent: dict = Depends(current_agent)):
    return {"email": agent.get("sub"), "name": agent.get("name"), "role": agent.get("role")}


@app.get("/analytics")
def analytics(agent: dict = Depends(require_real_agent)):
    """This agent's own performance, computed from their handled tickets.

    No seeded history: a new agent sees zeros, and the UI says so rather than
    drawing a chart that implies activity which never happened.
    """
    return compute_analytics(get_store().events(agent=agent.get("sub")))


@app.post("/analyze")
def analyze(req: AnalyzeRequest, agent: dict = Depends(current_agent)):
    try:
        c = coach()
        store = get_store()

        # Prior turns, before this message is recorded, so the analyser sees
        # the conversation that led here rather than an echo of the input.
        if req.ticket_id:
            history_text = store.history_text(req.ticket_id)
        else:
            history_text = "\n".join(f"{h.speaker}: {h.text}" for h in req.history)

        analysis = c.analyze_customer_message(req.message, history_text)

        # A customer writing in again reopens a resolved ticket. Their sentiment
        # on this follow-up is the "after" half of the shift metric — how they
        # sound having read the agent's reply.
        if req.ticket_id:
            if store.history(req.ticket_id):
                store.record_sentiment_after(req.ticket_id, analysis.get("sentiment"))
            store.reopen(req.ticket_id)
            store.add_message(req.ticket_id, "customer", req.message)
            history_text = store.history_text(req.ticket_id)

        order_id = clean_key(analysis.get("order_id"))
        issue_type = clean_key(analysis.get("issue_type"))

        # ---- keyed lookups: exact, by ID ----
        order = get_order(order_id)
        policy = get_policy(issue_type)
        # ---- retrieval: real BM25 ranking over the FAQ corpus ----
        # issue_type is a boost, not a filter; the whole corpus is searched.
        faqs = search_faqs(req.message, k=3, category=issue_type)

        # history_text already includes this turn when a ticket_id was given;
        # for a stateless call, append it so the drafter sees the full thread.
        if not req.ticket_id:
            history_text = (history_text + f"\ncustomer: {req.message}").strip()

        suggested = c.suggest_reply(req.message, history_text, order, policy, faqs)

        return {
            "sentiment": analysis["sentiment"],
            "urgency": analysis["urgency"],
            "escalation_risk": analysis["escalation_risk"],
            "key_issue": analysis["key_issue"],
            "order_id": order_id,
            "issue_type": issue_type,
            "order": order,
            "policy": policy,
            "faqs": faqs,
            "order_found": order is not None,
            "policy_found": policy is not None,
            "suggested_reply": suggested,
            "history": store.history(req.ticket_id) if req.ticket_id else [],
        }
    except ValueError as e:
        # Missing key etc.
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Analysis failed: {e}")


@app.get("/queue")
def queue(agent: dict = Depends(current_agent)):
    """The open backlog, ordered by a priority queue.

    Each ticket is analysed on its own (one model call per ticket, run
    concurrently), scored from that analysis, then pushed into a binary heap.
    Draining the heap gives the order an agent should work in.

    This is a third event — an agent opening the queue — so it is its own
    endpoint. It deliberately does NOT draft replies: that costs a second call
    per ticket and only matters once an agent picks one, which is what
    /analyze is for.
    """
    try:
        c = coach()
        store = get_store()
        resolved = store.resolved_ids()

        # A handled ticket leaves the queue. Analysing it anyway would burn a
        # model call per refresh to rank something nobody should work on.
        tickets = [t for t in get_tickets() if t["ticket_id"] not in resolved]

        def enrich(ticket: dict) -> dict:
            tid = ticket["ticket_id"]
            # Rate the whole thread — a follow-up reads differently from a
            # first contact, and the queue should reflect that.
            history_text = store.history_text(tid)
            analysis = c.analyze_customer_message(ticket["message"], history_text)
            score, parts = priority_score(analysis, ticket)
            return {
                **ticket,
                "turns": len(store.history(tid)),
                "sentiment": analysis.get("sentiment"),
                "urgency": analysis.get("urgency"),
                "escalation_risk": analysis.get("escalation_risk"),
                "key_issue": analysis.get("key_issue"),
                "order_id": clean_key(analysis.get("order_id")),
                "issue_type": clean_key(analysis.get("issue_type")),
                "priority_score": score,
                "score_parts": parts,
            }

        # One call per ticket; a serial loop would make the dashboard feel dead.
        with ThreadPoolExecutor(max_workers=min(8, len(tickets) or 1)) as pool:
            enriched = list(pool.map(enrich, tickets))

        q = TicketQueue()
        for t in enriched:
            q.push(t, t["priority_score"])

        ordered = q.drain()
        for rank, t in enumerate(ordered, start=1):
            t["rank"] = rank

        return {
            "tickets": ordered,
            "count": len(ordered),
            "resolved_count": len(resolved),
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Queue build failed: {e}")


@app.post("/evaluate")
def evaluate(req: EvaluateRequest, agent: dict = Depends(current_agent)):
    try:
        fb = coach().evaluate_agent_response(req.customer_message, req.agent_message)

        # Record the agent's turn and close the ticket. Scoring and resolving
        # are the same event here: the agent has answered and moved on.
        store = get_store()
        if req.ticket_id:
            store.add_message(req.ticket_id, "agent", req.agent_message)
            if req.resolve:
                store.resolve(req.ticket_id)
            # The dashboard is built from these rows. `sentiment_after` stays
            # None until (and unless) the customer writes back.
            store.record_handled(agent.get("sub"), req.ticket_id, {
                "tone_score": fb.tone_score,
                "empathy_score": fb.empathy_score,
                "clarity_score": fb.clarity_score,
                "coaching_tip": fb.coaching_tip,
                "sentiment_before": req.sentiment_before,
                "sentiment_after": None,
                "issue_type": req.issue_type,
                "escalation_risk": req.escalation_risk,
            })

        return {
            "tone_score": fb.tone_score,
            "empathy_score": fb.empathy_score,
            "clarity_score": fb.clarity_score,
            "coaching_tip": fb.coaching_tip,
            "resolved": bool(req.ticket_id and req.resolve),
            "history": store.history(req.ticket_id) if req.ticket_id else [],
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {e}")


@app.get("/tickets/{ticket_id}")
def ticket(ticket_id: str, agent: dict = Depends(current_agent)):
    """The stored thread for one ticket, plus whether it is still open."""
    store = get_store()
    return {
        "ticket_id": ticket_id,
        "history": store.history(ticket_id),
        "resolved": store.is_resolved(ticket_id),
    }


@app.post("/tickets/{ticket_id}/reopen")
def reopen_ticket(ticket_id: str, agent: dict = Depends(current_agent)):
    """Put a handled ticket back in the queue without a new customer message."""
    store = get_store()
    store.reopen(ticket_id)
    return {"ticket_id": ticket_id, "resolved": False}


@app.post("/tickets/reset")
def reset_tickets(agent: dict = Depends(current_agent)):
    """Clear all conversation history and reopen everything. Demo reset."""
    get_store().reset()
    return {"status": "reset"}
