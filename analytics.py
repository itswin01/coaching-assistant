"""
Support Coach — agent performance analytics.
============================================

Every number here is computed from events the agent actually generated. There
is no seeded history: a new install returns zeros and empty series, and the UI
says so rather than drawing a plausible-looking empty chart.

The headline metric is `sentiment_shift`. It is measurable rather than guessed:
when an agent replies we store how the customer sounded beforehand, and when
that customer writes again we store how they sound then. Comparing the two says
whether the reply calmed them or made things worse. It only covers tickets where
the customer wrote back, so `measured` is always reported alongside `handled` —
never quote the split without it.
"""

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

# Ordered worst -> best, so a move up the scale is an improvement.
SENTIMENT_RANK = {"negative": 0, "neutral": 1, "positive": 2}


def _day(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc).date().isoformat()
    except (ValueError, TypeError):
        return date.today().isoformat()


def _mean(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def compute(events: List[Dict[str, Any]], days: int = 14) -> Dict[str, Any]:
    """Turn raw handled-ticket events into the dashboard payload."""
    handled = [e for e in events if e.get("type") == "handled"]

    # ---- sentiment shift: calmed / unchanged / escalated ----
    shift = {"calmed": 0, "unchanged": 0, "escalated": 0}
    for e in handled:
        before, after = e.get("sentiment_before"), e.get("sentiment_after")
        if not before or not after:
            continue  # customer never wrote back — not measurable, not counted
        b, a = SENTIMENT_RANK.get(before, 1), SENTIMENT_RANK.get(after, 1)
        if a > b:
            shift["calmed"] += 1
        elif a < b:
            shift["escalated"] += 1
        else:
            shift["unchanged"] += 1
    measured = sum(shift.values())

    # ---- volume per day, zero-filled so the chart has a real time axis ----
    today = datetime.now(timezone.utc).date()
    window = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    per_day = Counter(_day(e["at"]) for e in handled)
    volume = [{"date": d, "count": per_day.get(d, 0)} for d in window]

    # ---- coaching scores, overall and per day ----
    tone = [e["tone_score"] for e in handled if e.get("tone_score") is not None]
    emp = [e["empathy_score"] for e in handled if e.get("empathy_score") is not None]
    clar = [e["clarity_score"] for e in handled if e.get("clarity_score") is not None]

    by_day: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"tone": [], "empathy": [], "clarity": []}
    )
    for e in handled:
        d = _day(e["at"])
        for key, field in (("tone", "tone_score"), ("empathy", "empathy_score"), ("clarity", "clarity_score")):
            if e.get(field) is not None:
                by_day[d][key].append(e[field])
    trend = [
        {
            "date": d,
            "tone": _mean(by_day[d]["tone"]) if d in by_day else None,
            "empathy": _mean(by_day[d]["empathy"]) if d in by_day else None,
            "clarity": _mean(by_day[d]["clarity"]) if d in by_day else None,
        }
        for d in window
    ]

    tips = Counter(e["coaching_tip"] for e in handled if e.get("coaching_tip"))

    return {
        "handled": len(handled),
        "sentiment_shift": {**shift, "measured": measured},
        "volume": volume,
        "today": per_day.get(today.isoformat(), 0),
        "busiest_day": max((v["count"] for v in volume), default=0),
        "scores": {
            "tone": _mean(tone),
            "empathy": _mean(emp),
            "clarity": _mean(clar),
            "overall": _mean(tone + emp + clar),
        },
        "score_trend": trend,
        "by_issue": dict(Counter(e.get("issue_type") or "unclassified" for e in handled)),
        "by_risk": dict(Counter(e.get("escalation_risk") or "unknown" for e in handled)),
        "top_tips": [{"tip": t, "count": n} for t, n in tips.most_common(3)],
    }
