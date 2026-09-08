"""
Support Coach — FAQ retrieval.
==============================

Real lexical retrieval over `faqs.json` using Okapi BM25 — the ranking function
behind Lucene and Elasticsearch. There are no hardcoded match percentages here:
every score comes from term statistics over the actual corpus.

What this is:
  - genuine ranked retrieval with IDF weighting and length normalisation
  - field-weighted (a hit in the title counts for more than one in the body)
  - query expansion over a small support-vocabulary synonym map

What this is NOT:
  - semantic / embedding search. BM25 matches words, not meaning. A query that
    shares no vocabulary with a FAQ will not retrieve it, however related it is.
    The synonym map below closes the most common gaps ("money back" -> refund);
    it is not a substitute for embeddings.

Swap point: `FaqIndex.search` is the only thing callers use. Replacing BM25 with
a vector index means reimplementing that one method.
"""

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# BM25 parameters. k1 controls term-frequency saturation, b controls how much
# document length is penalised. These are the standard Lucene defaults.
K1 = 1.5
B = 0.75

# Field weights — a term in the title is stronger evidence than one in the body.
FIELD_WEIGHTS = {"title": 3, "tags": 2, "body": 1}

# Two gates decide whether a hit is real, because neither works alone.
# MIN_SCORE is an absolute floor. MIN_SCORE_PER_TERM normalises by how many
# content words the customer actually used — without it a long off-topic
# message accumulates enough weak partial hits to clear any fixed floor, while
# a short legitimate one ("where is my money") falls under it.
MIN_SCORE = 3.0
MIN_SCORE_PER_TERM = 1.8

# The per-term floor is capped, because BM25 does not grow with query length —
# only *matching* terms add score. "...I need this sorted before my trip on
# Friday" adds seven words to a damaged-item report: +12.6 to an uncapped floor
# and nothing to the score, so the longer and more detailed the customer is, the
# harder it becomes to match. Past a handful of content words the query has
# already shown what it is about.
FLOOR_TERM_CAP = 5

# Synonym hits are weaker evidence than the customer's own words. Scoring them
# at parity lets a generic expansion term ("delivery") outrank a document that
# matched what was actually typed.
EXPANSION_WEIGHT = 0.45

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "about", "as",
    "by", "from", "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "them", "their", "have",
    "has", "had", "do", "does", "did", "can", "could", "will", "would", "should",
    "there", "here", "what", "when", "where", "who", "how", "why", "not", "no",
    "so", "up", "out", "get", "got", "just", "very", "please", "hi", "hello",
}

# Query expansion. Customers do not use the vocabulary the knowledge base is
# written in — this maps how people actually phrase things onto the words the
# FAQs use. Expansion applies to the query only, never to the documents.
SYNONYMS = {
    "money": ["refund"],
    "cash": ["refund"],
    "reimburse": ["refund"],
    "reimbursement": ["refund"],
    "repay": ["refund"],
    "parcel": ["package", "shipping", "delivery"],
    "package": ["shipping", "delivery"],
    "posted": ["shipping"],
    "postage": ["shipping"],
    "courier": ["carrier", "shipping"],
    "arrive": ["delivery", "shipping"],
    "arrived": ["delivery", "shipping"],
    "late": ["delay", "delayed"],
    "waiting": ["delay", "delayed"],
    "broken": ["damaged", "faulty"],
    "cracked": ["damaged"],
    "smashed": ["damaged"],
    "shattered": ["damaged"],
    "crushed": ["damaged"],
    "dented": ["damaged"],
    "faulty": ["defect", "technical"],
    "defective": ["defect", "technical"],
    "dead": ["power", "technical"],
    "boot": ["power", "start"],
    "login": ["sign", "account"],
    "signin": ["sign", "account"],
    "password": ["account", "reset"],
    "locked": ["account", "locked"],
    "charged": ["billing", "charge"],
    "billed": ["billing", "charge"],
    "invoice": ["billing", "receipt"],
    "swap": ["exchange", "returns"],
    "replace": ["replacement"],
    "cancel": ["cancellation"],
    "voucher": ["promo", "discount"],
    "coupon": ["promo", "discount"],
    "code": ["promo", "discount"],
    "guarantee": ["warranty"],
    "repair": ["technical", "warranty"],
}

_WORD = re.compile(r"[a-z0-9]+")

# Order IDs are lookup keys, not retrieval terms — `get_order` already handles
# them exactly. Left in the query they are pure noise: "WORD-88221" splits into
# "word" + "88221" and drags in whichever FAQ happens to document the ID format.
_ORDER_ID = re.compile(r"\b[a-z]{2,6}-\d{3,8}\b", re.IGNORECASE)


def strip_keys(text: str) -> str:
    """Remove order-ID-shaped tokens before retrieval."""
    return _ORDER_ID.sub(" ", text or "")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, strip plurals."""
    out = []
    for w in _WORD.findall((text or "").lower()):
        if w in STOPWORDS or len(w) < 2:
            continue
        # Crude but effective singularisation: "refunds" and "refund" must match.
        if len(w) > 3 and w.endswith("es") and not w.endswith("ses"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def build_query(query: str) -> Tuple[Dict[str, float], int]:
    """Turn a raw message into weighted query terms.

    Returns the term->weight map and the count of the customer's own distinct
    content words, which the per-term threshold in `search` normalises against.
    Documents are never expanded — expansion is a query-side technique only.
    """
    base = tokenize(strip_keys(query))
    weights: Dict[str, float] = {t: 1.0 for t in base}
    for t in base:
        for syn in SYNONYMS.get(t, []):
            for st in tokenize(syn):
                # Never let an expansion downgrade a term the customer typed.
                weights.setdefault(st, EXPANSION_WEIGHT)
    return weights, len(set(base))


class FaqIndex:
    """An in-memory BM25 index over the FAQ corpus."""

    def __init__(self, faqs: List[Dict[str, Any]]):
        self.faqs = faqs
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        self.df: Counter = Counter()

        for faq in faqs:
            tokens: List[str] = []
            # Field weighting is done by repeating the field's tokens, which is
            # equivalent to scaling that field's term frequency.
            for field, weight in FIELD_WEIGHTS.items():
                value = faq.get(field, "")
                text = " ".join(value) if isinstance(value, list) else str(value)
                tokens.extend(tokenize(text) * weight)
            self.doc_tokens.append(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_len.append(len(tokens))
            for term in freqs:
                self.df[term] += 1

        self.n = len(faqs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        """Standard BM25 inverse document frequency, always positive."""
        n_q = self.df.get(term, 0)
        return math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))

    def _score(self, weights: Dict[str, float], i: int) -> float:
        freqs = self.doc_freqs[i]
        dl = self.doc_len[i]
        score = 0.0
        for term, w in weights.items():
            f = freqs.get(term, 0)
            if not f:
                continue
            denom = f + K1 * (1 - B + B * (dl / self.avgdl if self.avgdl else 1))
            score += w * self.idf(term) * (f * (K1 + 1)) / denom
        return score

    def search(
        self,
        query: str,
        k: int = 3,
        category: Optional[str] = None,
        min_score: float = MIN_SCORE,
    ) -> List[Dict[str, Any]]:
        """Rank the corpus against `query` and return the top k above threshold.

        `category` (the analyzer's issue_type) is a *boost*, not a filter — the
        whole corpus is always searched, so a billing FAQ can still win on a
        ticket classified as a refund. Filtering would hide the corpus's range.
        """
        weights, n_terms = build_query(query)
        if not weights:
            return []

        # Scales with the query so a one-word question is not held to the same
        # absolute bar as a paragraph — but capped, see FLOOR_TERM_CAP.
        floor = max(min_score, MIN_SCORE_PER_TERM * min(max(n_terms, 1), FLOOR_TERM_CAP))

        scored: List[Tuple[float, int]] = []
        for i in range(self.n):
            s = self._score(weights, i)
            if s <= 0:
                continue
            if category and self.faqs[i].get("category") == category:
                s *= 1.15
            scored.append((s, i))

        scored.sort(key=lambda x: (-x[0], self.faqs[x[1]]["id"]))

        results = []
        for s, i in scored[:k]:
            if s < floor:
                break
            faq = self.faqs[i]
            results.append({
                "id": faq["id"],
                "title": faq["title"],
                "body": faq["body"],
                "category": faq["category"],
                # The real BM25 score, rounded for display. Deliberately NOT a
                # percentage — BM25 is unbounded and a "%" would invent precision.
                "score": round(s, 2),
            })
        return results


def load_faqs(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "faqs.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Built once at import; the corpus is static and small enough to hold in memory.
_INDEX: Optional[FaqIndex] = None


def get_index() -> FaqIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = FaqIndex(load_faqs())
    return _INDEX


def search_faqs(query: str, k: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Public entry point. Returns [] when nothing clears the score threshold."""
    return get_index().search(query, k=k, category=category)
