"""
Support Coach — shared coaching logic.
=====================================

Single source of truth for the coach: the stub fact layer and the model calls.
`app.py` (FastAPI) and `support_coach.ipynb` both use this module, so the
prompts and the stub data exist in exactly one place.

Nothing here knows about HTTP. Swap `get_order` / `get_policy` / `get_tickets`
for real data sources and neither caller changes.

FAQ retrieval lives in `retrieval.py` (real BM25 over faqs.json), not here.
"""

import os
import json
import time
import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


# ------------------------------------------------------------------ #
#  Data models                                                       #
# ------------------------------------------------------------------ #

@dataclass
class Message:
    """One message in the conversation."""
    speaker: str
    text: str


@dataclass
class ConversationState:
    """Full conversation plus the latest analysis of the customer."""
    history: List[Message] = field(default_factory=list)

    sentiment: str = "unknown"
    urgency: str = "unknown"
    escalation_risk: str = "unknown"
    key_issue: str = ""

    # Lookup keys extracted from the customer message
    order_id: Optional[str] = None
    issue_type: Optional[str] = None

    def add_message(self, speaker: str, text: str):
        self.history.append(Message(speaker=speaker, text=text))


@dataclass
class CoachingFeedback:
    """Scores + tip for an agent reply."""
    tone_score: int
    empathy_score: int
    clarity_score: int
    coaching_tip: str


# ------------------------------------------------------------------ #
#  Fact layer (STUBS) — swap get_order / get_policy for real sources  #
# ------------------------------------------------------------------ #

ORDERS: Dict[str, Dict[str, Any]] = {
    "WORD-88221": {
        "order_id": "WORD-88221",
        "product": 'Pro Laptop 16"',
        "status": "refund_pending",
        "days_since_refund_request": 20,
        "refund_amount": "$1,499.00",
        "customer_tier": "Pro",
    },
    "WORD-10233": {
        "order_id": "WORD-10233",
        "product": "Wireless Mouse",
        "status": "shipped",
        "days_since_order": 3,
        "tracking": "TRK-556677",
        "customer_tier": "Standard",
    },
}

POLICIES: Dict[str, str] = {
    "refund": "Approved refunds are processed within 5-7 business days. "
              "Refunds pending beyond 10 business days should be escalated to billing.",
    "shipping": "Standard shipping is 3-5 business days; express is 1-2. "
                "Tracking is emailed once the order ships.",
    "damaged": "For damaged items, offer a replacement or full refund. "
               "Request a photo only if needed for a carrier claim; never require it before helping.",
    "technical": "Route hardware faults under warranty to technical support for a diagnostic. "
                 "Warranty covers manufacturing defects for 12 months.",
    "account": "Verify identity before any account change. Never ask for full passwords.",
}

VALID_ISSUE_TYPES = list(POLICIES.keys())


def get_order(order_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Direct keyed lookup. Swap for a real DB/API call later."""
    if not order_id:
        return None
    return ORDERS.get(order_id.strip().upper())


def get_policy(issue_type: Optional[str]) -> Optional[str]:
    """Direct keyed lookup. Swap for a real policy store later."""
    if not issue_type:
        return None
    return POLICIES.get(issue_type.strip().lower())


# ------------------------------------------------------------------ #
#  Ticket queue                                                      #
# ------------------------------------------------------------------ #

# ---- STUB: the open ticket backlog, one message per waiting customer ----
# Same status as ORDERS/POLICIES: hand-written. Swap `get_tickets` for a real
# helpdesk queue (Zendesk/Freshdesk/Jira view) and nothing downstream changes.
TICKETS: List[Dict[str, Any]] = [
    {
        "ticket_id": "TCK-4471",
        "customer": "R. Mehta",
        "customer_tier": "Pro",
        "waiting_minutes": 1740,
        "message": "I have been waiting for my refund on order WORD-88221 for 20 days! "
                   "This is ridiculous. I want my money back immediately!",
    },
    {
        "ticket_id": "TCK-4468",
        "customer": "S. Okonkwo",
        "customer_tier": "Standard",
        "waiting_minutes": 95,
        "message": "Where is my package for order WORD-10233? It was supposed to arrive already.",
    },
    {
        "ticket_id": "TCK-4472",
        "customer": "L. Fontaine",
        "customer_tier": "Pro",
        "waiting_minutes": 22,
        "message": "The laptop arrived with a cracked screen. The box was crushed on one corner. "
                   "I need this sorted before my trip on Friday.",
    },
    {
        "ticket_id": "TCK-4460",
        "customer": "D. Whitfield",
        "customer_tier": "Standard",
        "waiting_minutes": 2880,
        "message": "Hi, could you tell me how long express shipping usually takes? No rush, just planning ahead.",
    },
    {
        "ticket_id": "TCK-4475",
        "customer": "A. I'lic",
        "customer_tier": "Pro",
        "waiting_minutes": 8,
        "message": "My laptop won't turn on at all after the last update. Nothing on screen. "
                   "I use this for work and I'm completely stuck.",
    },
    {
        "ticket_id": "TCK-4463",
        "customer": "K. Nakamura",
        "customer_tier": "Standard",
        "waiting_minutes": 610,
        "message": "I can't sign in to my account and the password reset email never arrives. "
                   "Can you help me get back in?",
    },
]

# Weights for the priority score. Escalation risk dominates, then urgency —
# a calm customer who will churn still outranks a shouty one who won't.
RISK_WEIGHT = {"high": 1.0, "medium": 0.55, "low": 0.15}
URGENCY_WEIGHT = {"high": 1.0, "medium": 0.55, "low": 0.15}
SENTIMENT_WEIGHT = {"negative": 1.0, "neutral": 0.4, "positive": 0.0}
TIER_WEIGHT = {"pro": 1.0, "standard": 0.45}

# Points available per factor. Sums to 100.
POINTS = {"risk": 40, "urgency": 30, "sentiment": 15, "tier": 10, "age": 5}

# Waiting time saturates at 48h — past that it stops out-weighting real risk.
AGE_SATURATES_AT_MINUTES = 48 * 60


def get_tickets() -> List[Dict[str, Any]]:
    """Direct lookup of the open backlog. Swap for a real helpdesk query."""
    return [dict(t) for t in TICKETS]


def priority_score(analysis: Dict[str, Any], ticket: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """Turn one analysis + ticket into a 0-100 priority and its breakdown.

    Every input except tier and waiting time comes from the model's analysis of
    the actual message, so the ordering reflects what the customer wrote — not
    keyword matching.
    """
    risk = RISK_WEIGHT.get(str(analysis.get("escalation_risk", "")).lower(), 0.15)
    urgency = URGENCY_WEIGHT.get(str(analysis.get("urgency", "")).lower(), 0.15)
    sentiment = SENTIMENT_WEIGHT.get(str(analysis.get("sentiment", "")).lower(), 0.4)
    tier = TIER_WEIGHT.get(str(ticket.get("customer_tier", "")).lower(), 0.45)
    age = min(ticket.get("waiting_minutes", 0) / AGE_SATURATES_AT_MINUTES, 1.0)

    parts = {
        "risk": round(POINTS["risk"] * risk),
        "urgency": round(POINTS["urgency"] * urgency),
        "sentiment": round(POINTS["sentiment"] * sentiment),
        "tier": round(POINTS["tier"] * tier),
        "age": round(POINTS["age"] * age),
    }
    return sum(parts.values()), parts


class TicketQueue:
    """A real max-priority queue over tickets, backed by a binary heap.

    `heapq` is a min-heap, so the sort key is negated. The key is
    (-score, -waiting_minutes, seq): highest score first, then longest-waiting
    first, then insertion order — so ties never reorder unpredictably between
    calls, which matters when an agent is working down a live list.
    """

    def __init__(self):
        self._heap: List[Tuple[int, int, int, Dict[str, Any]]] = []
        self._counter = itertools.count()

    def push(self, ticket: Dict[str, Any], score: int) -> None:
        heapq.heappush(
            self._heap,
            (-score, -ticket.get("waiting_minutes", 0), next(self._counter), ticket),
        )

    def pop(self) -> Optional[Dict[str, Any]]:
        """Remove and return the highest-priority ticket, or None if empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[3]

    def peek(self) -> Optional[Dict[str, Any]]:
        return self._heap[0][3] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def drain(self) -> List[Dict[str, Any]]:
        """Pop everything in priority order. Consumes the queue."""
        out = []
        while self._heap:
            out.append(self.pop())
        return out


def format_history(history: Optional[List[Message]]) -> str:
    """Render a Message list as the plain transcript the prompt expects."""
    if not history:
        return ""
    return "\n".join(f"{m.speaker}: {m.text}" for m in history)


def clean_key(value: Any) -> Optional[str]:
    """Normalize a lookup key from the model's JSON: '', 'null', None -> None."""
    if not value:
        return None
    if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
        return None
    return value


# ------------------------------------------------------------------ #
#  AI Coach                                                          #
# ------------------------------------------------------------------ #

class AICoach:
    """The three model calls: analyze, suggest a grounded reply, score a reply.

    Runs on Groq's chat-completions API (OpenAI-shaped). Swapping providers
    means changing `_call` and `__init__` only — the prompts and the three
    public methods are provider-agnostic.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        # Imported here so this module can be imported (and the stub data used)
        # without the groq package installed.
        from groq import Groq
        self.client = Groq(api_key=api_key)
        # openai/gpt-oss-120b: follows the JSON contracts below reliably and is
        # fast enough for a per-message loop. Override with GROQ_MODEL —
        # openai/gpt-oss-20b is quicker if the queue feels slow.
        # Check availability with GET /openai/v1/models before changing this;
        # Groq retires model IDs faster than the docs page is updated.
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # ---------- helpers ----------

    def _extract_text(self, response) -> str:
        """Pull the assistant text out of a chat-completions response."""
        return (response.choices[0].message.content or "").strip()

    def _call(self, prompt: str, max_tokens: int = 400, retries: int = 2,
              json_mode: bool = False) -> str:
        """One API call with simple retry so a transient error doesn't kill the session.

        `json_mode` turns on Groq's JSON object mode. Open-weight models tend to
        wrap JSON in prose without it, and the callers that need JSON already
        state that explicitly in the prompt, which is what the mode requires.
        """
        last_err = None
        # Reasoning models (gpt-oss, qwen3) spend max_completion_tokens on thinking
        # before they emit anything. At a tight budget they return an EMPTY string,
        # which JSON mode then rejects with json_validate_failed. Keeping effort low
        # and the budget generous is what stops that.
        use_reasoning = True
        for attempt in range(retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens,
                    "temperature": 0.3,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if use_reasoning:
                    kwargs["reasoning_effort"] = "low"
                    # "raw" is rejected alongside JSON mode; hidden keeps the
                    # thinking out of the content we parse.
                    kwargs["reasoning_format"] = "hidden"
                resp = self.client.chat.completions.create(**kwargs)
                return self._extract_text(resp)
            except Exception as e:  # noqa: BLE001 - demo-level robustness
                last_err = e
                # Non-reasoning models reject reasoning_*; drop them and retry so
                # GROQ_MODEL can point at either kind.
                if use_reasoning and "reasoning" in str(e).lower():
                    use_reasoning = False
                    continue
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"API call failed after {retries + 1} attempts: {last_err}")

    def _parse_json(self, text: str) -> dict:
        """Convert the model's response to a dict, tolerating code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned by the model: {e}\nRaw response:\n{text}")

    # ---------- 1. analyze ----------

    def analyze_customer_message(self, customer_message: str, history_text: str = "") -> dict:
        # Prior turns change the reading of a message completely. "How much
        # longer?" is a mild question in isolation and an escalation on the
        # third ask after a broken promise. Rate the thread, not the sentence.
        history_block = (
            f"""Earlier in this conversation (oldest first):
{history_text}

"""
            if history_text.strip()
            else ""
        )
        prompt = f"""You are an AI customer-support risk analyzer. Your ratings are used
to rank a queue, so they only have value if they discriminate between tickets.

{history_block}Latest customer message:
{customer_message}

Rate against these definitions. Do not grade on a curve of politeness — a calm
customer who cannot work is a bigger problem than a rude one who is merely waiting.

escalation_risk — the risk THIS customer escalates, cancels, or complains publicly:
- high:   already angry AND has a concrete overdue grievance — an explicit demand,
          a threat to cancel/chargeback/review, a repeat contact, or a broken promise.
- medium: a legitimate unresolved problem and visible frustration, but no demand
          to escalate yet.
- low:    routine question, minor issue, or a cooperative tone.

urgency — how fast this must be answered:
- high:   the customer is blocked right now (cannot use the product, locked out)
          or states a hard deadline.
- medium: real inconvenience with time pressure, but a workaround or slack exists.
- low:    informational, or the customer says there is no rush.

sentiment — the tone of the message itself: positive, neutral, or negative.

Calibration: in a normal queue most tickets are low or medium. Reserve "high" for
cases that clearly meet the bar above. If you mark everything high, the ranking
is worthless.

If there is earlier conversation above, rate the thread as a whole: repeated
asking, a commitment already made and missed, or frustration building across
turns all raise escalation_risk even when the latest message alone reads calmly.

Also extract the lookup keys:
- "order_id": the order id if the customer mentions one (e.g. "WORD-88221"), else null.
- "issue_type": EXACTLY ONE of {VALID_ISSUE_TYPES}, or null if none fit.

Return ONLY valid JSON, exactly this structure:
{{
    "sentiment": "positive|neutral|negative",
    "urgency": "low|medium|high",
    "escalation_risk": "low|medium|high",
    "key_issue": "short description of the main issue",
    "order_id": "WORD-88221 or null",
    "issue_type": "one of {VALID_ISSUE_TYPES} or null"
}}

Do not add explanations outside the JSON."""
        return self._parse_json(self._call(prompt, max_tokens=1024, json_mode=True))

    # ---------- 2. suggest reply (grounded) ----------

    def suggest_reply(
        self,
        customer_message: str,
        history_text: str = "",
        order: Optional[Dict[str, Any]] = None,
        policy: Optional[str] = None,
        faqs: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Facts block. If we have no facts, we say so explicitly so the model
        # does NOT invent policy or order details.
        order_text = json.dumps(order, indent=2) if order else "No order record found."
        policy_text = policy if policy else "No specific policy found for this issue."
        # Retrieved knowledge-base articles (real BM25 hits from retrieval.py).
        # These join the facts block, so the same "state nothing that is not
        # here" rule governs them.
        if faqs:
            faq_text = "\n".join(f"{f['id']} — {f['title']}: {f['body']}" for f in faqs)
        else:
            faq_text = "No knowledge-base article matched this message."

        prompt = f"""You are an expert customer-support agent drafting a reply for a human agent to send.

Customer message:
{customer_message}

Previous conversation:
{history_text}

KNOWN FACTS (the only facts you may state as company policy or order status):
--- Order record ---
{order_text}
--- Relevant policy ---
{policy_text}
--- Retrieved knowledge-base articles ---
{faq_text}

Rules:
- Acknowledge the concern and show empathy (this drives tone).
- You MAY state specific numbers/status ONLY if they appear in KNOWN FACTS above.
- This includes procedural detail: do not invent durations, step counts, button
  timings, or instructions that are not written in KNOWN FACTS. If an article
  says "force restart" without saying how, say "force restart" without saying how.
- If a fact the customer wants is NOT in KNOWN FACTS, say you will check rather than inventing it.
- Do not promise anything the policy does not support.
- Write the message body only. No greeting placeholder such as [Customer Name],
  no sign-off placeholder — a human agent sends this from their own account.
- Keep it concise. Return only the reply text."""
        return self._call(prompt, max_tokens=350).strip()

    # ---------- 3. evaluate agent reply ----------

    def evaluate_agent_response(
        self, customer_message: str, agent_message: str
    ) -> CoachingFeedback:
        prompt = f"""You are an AI customer-support coach.

Customer message:
{customer_message}

Agent response:
{agent_message}

Score the agent 1-10 for Tone, Empathy, Clarity, then give ONE concrete coaching tip.

Return ONLY valid JSON, exactly:
{{
    "tone_score": 1,
    "empathy_score": 1,
    "clarity_score": 1,
    "coaching_tip": "one concrete coaching tip"
}}

No text outside the JSON."""
        result = self._parse_json(self._call(prompt, max_tokens=1024, json_mode=True))
        return CoachingFeedback(
            tone_score=int(result["tone_score"]),
            empathy_score=int(result["empathy_score"]),
            clarity_score=int(result["clarity_score"]),
            coaching_tip=result["coaching_tip"],
        )
