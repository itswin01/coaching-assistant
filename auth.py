"""
Support Coach — agent authentication.
=====================================

Demo-grade, server-enforced sign-in. "Server-enforced" is the important half:
the API itself rejects unauthenticated calls, so hiding the UI is not what keeps
customer queries private. A frontend-only gate would be a facade.

What this does properly:
  - passwords stored as PBKDF2-SHA256 with a per-password salt, never plaintext
  - constant-time hash comparison
  - HMAC-signed session tokens with an expiry, verified on every request

What this deliberately is NOT — say so plainly rather than implying otherwise:
  - there is no TLS here. Over plain HTTP a password is readable on the wire.
  - no rate limiting or lockout, so it is brute-forceable
  - no signup, password reset, or rotation — accounts are seeded, by design
    (this is company-to-company: agents are provisioned, not self-registered)
  - the signing secret defaults to a per-process random value, so every restart
    invalidates outstanding tokens. Set SUPPORT_COACH_SECRET to keep sessions
    across restarts.

Do not put this on a public network as-is.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional

AGENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents.json")

PBKDF2_ITERATIONS = 200_000
TOKEN_TTL_SECONDS = 12 * 60 * 60  # a working day

# Seeded on first run. Documented in the README — these are demo accounts, not
# secrets, and the file they land in is gitignored.
SEED_AGENTS = [
    {"email": "agent@support.local", "name": "A. Rivera", "password": "demo1234"},
    {"email": "lead@support.local", "name": "M. Chen", "password": "demo1234"},
]

_SECRET = os.getenv("SUPPORT_COACH_SECRET") or secrets.token_hex(32)


# ------------------------------------------------------------------ #
#  Password hashing                                                  #
# ------------------------------------------------------------------ #

def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(dk).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, b64salt, b64hash = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), base64.b64decode(b64salt), int(iters)
        )
        # Constant time — a plain == leaks how much of the hash matched.
        return hmac.compare_digest(dk, base64.b64decode(b64hash))
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------------ #
#  Agent records                                                     #
# ------------------------------------------------------------------ #

def _seed_if_missing() -> None:
    if os.path.exists(AGENTS_FILE):
        return
    agents = {
        a["email"]: {"name": a["name"], "password": hash_password(a["password"])}
        for a in SEED_AGENTS
    }
    with open(AGENTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"agents": agents}, f, indent=2)
    print("\n  Seeded agent accounts (demo credentials):")
    for a in SEED_AGENTS:
        print(f"    {a['email']}  /  {a['password']}")
    print(f"  Stored hashed in {AGENTS_FILE}\n")


def load_agents() -> Dict[str, Dict[str, str]]:
    _seed_if_missing()
    try:
        with open(AGENTS_FILE, encoding="utf-8") as f:
            return json.load(f).get("agents", {})
    except (OSError, json.JSONDecodeError):
        return {}


def authenticate(email: str, password: str) -> Optional[Dict[str, str]]:
    """Return the agent record on success, None otherwise.

    The same None is returned for an unknown email and a wrong password, and a
    dummy hash is verified when the email is unknown, so response timing does
    not reveal which addresses exist.
    """
    agents = load_agents()
    record = agents.get((email or "").strip().lower())
    if record is None:
        hash_password(password or "")  # equalise timing
        return None
    if not verify_password(password or "", record["password"]):
        return None
    return {"email": email.strip().lower(), "name": record.get("name", "Agent")}


# ------------------------------------------------------------------ #
#  Session tokens                                                    #
# ------------------------------------------------------------------ #

def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(email: str, name: str, role: str = "agent") -> str:
    payload = {
        "sub": email,
        "name": name,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(_SECRET.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Verify signature and expiry. Returns the payload or None."""
    if not token or "." not in token:
        return None
    body, _, sig = token.partition(".")
    expected = _b64(hmac.new(_SECRET.encode(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


def make_demo_token() -> str:
    """Demo mode is unauthenticated by design and says so.

    It still carries a token so there is one auth path through the API rather
    than an unauthenticated bypass branch that could be reached accidentally.
    Role "demo" cannot read agent analytics.
    """
    return make_token("demo@local", "Demo", role="demo")
