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
    POST /analyze   { "message": "...", "history": [{speaker,text}, ...] }
        -> analysis + looked-up facts + grounded suggested reply
    POST /evaluate  { "customer_message": "...", "agent_message": "..." }
        -> tone/empathy/clarity scores + coaching tip
    GET  /health
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from coach import (
    AICoach,
    TicketQueue,
    clean_key,
    get_faqs,
    get_order,
    get_policy,
    get_tickets,
    priority_score,
)

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


class HistoryItem(BaseModel):
    speaker: str
    text: str


class AnalyzeRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []


class EvaluateRequest(BaseModel):
    customer_message: str
    agent_message: str


@app.get("/health")
def health():
    return {"status": "ok", "key_set": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        c = coach()
        analysis = c.analyze_customer_message(req.message)

        order_id = clean_key(analysis.get("order_id"))
        issue_type = clean_key(analysis.get("issue_type"))

        # ---- fact lookups (keyed, not searched) ----
        order = get_order(order_id)
        policy = get_policy(issue_type)
        faqs = get_faqs(issue_type)

        history_text = "\n".join(f"{h.speaker}: {h.text}" for h in req.history)
        history_text += f"\ncustomer: {req.message}"

        suggested = c.suggest_reply(req.message, history_text, order, policy)

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
        }
    except ValueError as e:
        # Missing key etc.
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Analysis failed: {e}")


@app.get("/queue")
def queue():
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
        tickets = get_tickets()

        def enrich(ticket: dict) -> dict:
            analysis = c.analyze_customer_message(ticket["message"])
            score, parts = priority_score(analysis, ticket)
            return {
                **ticket,
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

        return {"tickets": ordered, "count": len(ordered)}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Queue build failed: {e}")


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        fb = coach().evaluate_agent_response(req.customer_message, req.agent_message)
        return {
            "tone_score": fb.tone_score,
            "empathy_score": fb.empathy_score,
            "clarity_score": fb.clarity_score,
            "coaching_tip": fb.coaching_tip,
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {e}")
