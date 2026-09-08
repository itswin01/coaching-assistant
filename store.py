"""
Support Coach — conversation and ticket state.
==============================================

Holds what the stateless request path cannot: the message history per ticket
and whether a ticket is still open.

Deliberately small. This is a JSON file behind a lock, not a database — no
schema, no migrations, no ORM. It exists because a queue that cannot forget a
handled ticket is not a queue, and an analyzer that cannot see the previous
turn misreads every follow-up.

Swap point: `Store` is the only thing callers touch. Backing it with Postgres
or a helpdesk API means reimplementing this one class.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Per-ticket conversation history plus open/resolved state.

    Every public method takes the lock: /queue analyses tickets on a thread
    pool, so reads and writes genuinely overlap.
    """

    def __init__(self, path: str = STATE_FILE):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"conversations": {}, "resolved": {}}
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            self._data["conversations"] = loaded.get("conversations", {})
            self._data["resolved"] = loaded.get("resolved", {})
        except (OSError, json.JSONDecodeError):
            # A corrupt state file must not stop the server booting — the
            # tickets themselves come from coach.TICKETS and are unaffected.
            pass

    def _save_locked(self) -> None:
        """Write state. Caller already holds the lock."""
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)  # atomic, so a crash cannot truncate it
        except OSError:
            pass

    # ---------- conversation history ----------

    def add_message(self, ticket_id: str, speaker: str, text: str) -> None:
        """Append one turn. `speaker` is "customer" or "agent"."""
        if not ticket_id:
            return
        with self._lock:
            convo = self._data["conversations"].setdefault(ticket_id, [])
            convo.append({"speaker": speaker, "text": text, "at": _now()})
            self._save_locked()

    def history(self, ticket_id: Optional[str]) -> List[Dict[str, str]]:
        if not ticket_id:
            return []
        with self._lock:
            return list(self._data["conversations"].get(ticket_id, []))

    def history_text(self, ticket_id: Optional[str], limit: int = 12) -> str:
        """The transcript as the prompts want it, most recent `limit` turns."""
        turns = self.history(ticket_id)[-limit:]
        return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)

    # ---------- open / resolved ----------

    def resolve(self, ticket_id: str) -> None:
        """Mark handled. Resolved tickets drop out of the queue."""
        if not ticket_id:
            return
        with self._lock:
            self._data["resolved"][ticket_id] = _now()
            self._save_locked()

    def reopen(self, ticket_id: str) -> None:
        """Put a ticket back in the queue — a customer wrote in again."""
        if not ticket_id:
            return
        with self._lock:
            if self._data["resolved"].pop(ticket_id, None) is not None:
                self._save_locked()

    def is_resolved(self, ticket_id: str) -> bool:
        with self._lock:
            return ticket_id in self._data["resolved"]

    def resolved_ids(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._data["resolved"])

    def reset(self) -> None:
        """Clear all state — for demo resets and tests."""
        with self._lock:
            self._data = {"conversations": {}, "resolved": {}}
            self._save_locked()


_STORE: Optional[Store] = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE
