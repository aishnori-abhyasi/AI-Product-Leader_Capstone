"""
RBAC for customers and staff.

Principles
  * Least privilege + separation of duties: no staff role can do everything.
      - admin can manage users but CANNOT read customer cases or the audit trail
      - compliance_officer can read audit + consents but NOT scores/explanations
      - support sees masked cases and can review, but not model internals
      - risk_analyst sees explanations and fairness reports, but not consent/audit admin
  * Customers can only touch their own records (object-level check done in the route: owner == token.sub).
  * Every denial is written to the tamper-evident audit log.
  * Short-lived signed tokens (15 min) with a revocation list.
Production: use your IdP (OIDC) + MFA for staff, rotate secrets in a KMS. Passwords here use scrypt.
"""
import hashlib, hmac, os, secrets, time, uuid
from functools import wraps

import jwt
from flask import g, jsonify, request

SECRET = os.getenv("APP_JWT_SECRET") or secrets.token_hex(32)       # random per process if unset (dev only)
TOKEN_TTL = 900
STORE = None            # set by init()

PERMISSIONS = {
    "customer": {"self:analyse", "self:read", "self:contest", "self:consent", "self:export", "self:erase", "self:grievance", "self:correct", "self:notifications"},
    "support": {"case:read_masked", "case:review", "grievance:read", "grievance:resolve", "correction:read", "correction:process", "incident:declare"},
    "risk_analyst": {"case:read_masked", "case:review", "explain:read_case", "model:read_report", "fairness:read", "incident:declare"},
    "compliance_officer": {"audit:read", "consent:read", "grievance:read", "grievance:resolve",
                           "fairness:read", "model:read_report", "erasure:process", "correction:read",
                           "incident:declare", "incident:read", "incident:manage", "system:pause", "system:revoke_sessions"},
    "admin": {"user:manage", "incident:declare", "system:pause", "system:revoke_sessions"},
}
REVOKED = set()
NOT_BEFORE_MS = 0                  # tokens issued at or before this instant are invalid (incident response)

def init(store): global STORE; STORE = store

def hash_password(pw, salt=None):
    salt = salt or secrets.token_bytes(16)
    return salt.hex() + ":" + hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1).hex()

def check_password(pw, stored):
    salt, _, h = stored.partition(":")
    cand = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1).hex()
    return hmac.compare_digest(cand, h)

def issue_token(sub, role, ttl=TOKEN_TTL):
    t = int(time.time())
    return jwt.encode({"sub": sub, "role": role, "iat": t, "ims": int(time.time() * 1000), "exp": t + ttl, "jti": uuid.uuid4().hex}, SECRET, algorithm="HS256")

def _claims():
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer "): return None
    try:
        c = jwt.decode(h[7:], SECRET, algorithms=["HS256"], options={"require": ["exp", "sub", "role", "jti"]})
    except jwt.PyJWTError:
        return None
    return None if (c["jti"] in REVOKED or c.get("ims", 0) <= NOT_BEFORE_MS) else c

def require(*perms):
    """Route decorator: caller must hold ALL listed permissions."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            c = _claims()
            if not c:
                return jsonify(error="authentication required"), 401
            if not set(perms) <= PERMISSIONS.get(c["role"], set()):
                if STORE: STORE.audit(c["sub"], c["role"], "access_denied", request.path, "denied", needed=list(perms))
                return jsonify(error="forbidden"), 403
            g.claims = c
            return fn(*a, **kw)
        return wrapper
    return deco

def revoke_all_sessions():
    """Invalidate every token issued so far (customers and staff). Everyone must log in again."""
    global NOT_BEFORE_MS
    NOT_BEFORE_MS = int(time.time() * 1000)

def revoke_current():
    if getattr(g, "claims", None): REVOKED.add(g.claims["jti"])

class Limiter:
    """Tiny in-memory sliding-window limiter. Use Redis (or your API gateway) in production."""
    def __init__(self): self.hits = {}
    def blocked(self, key, n, window):
        t = time.time(); q = [x for x in self.hits.get(key, []) if t - x < window]
        self.hits[key] = q
        if len(q) >= n: return True
        q.append(t); return False
