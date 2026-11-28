"""
Privacy, consent and audit primitives, designed around the DPDP Act 2023 / DPDP Rules 2025,
RBI Digital Lending Directions 2025 and the Account Aggregator "one purpose, one consent" rule.
Standard library only.  NOT legal advice - have counsel sign off the notice text and retention.

What lives here
  * pseudonym()/mask_vpa()   the raw UPI ID is NEVER stored or logged; only an HMAC pseudonym
  * Store.grant/revoke_consent   purpose-limited, time-limited, revocable consent ledger
  * Store.audit()            hash-chained, tamper-evident audit trail (no PII in entries)
  * Store.export/erase/purge_expired   access, erasure and retention (storage limitation)
  * Store grievance table    redress channel
"""
import hashlib, hmac, json, os, re, sqlite3, threading, uuid
from datetime import datetime, timedelta, timezone

PEPPER = os.getenv("APP_PEPPER", "dev-only-pepper-change-me")        # keep in a KMS/HSM in production
NOTICE_VERSION = "2026-11-v1"
PURPOSE = "suggest_government_schemes_and_savings"                   # ONE purpose per consent
DATA_TYPES = ["UPI-linked bank account transactions (amounts, dates, counter-party codes)"]
FI_DAYS, DATA_LIFE_DAYS = 90, 30            # window we ask for / days we keep derived results
VPA_RE = re.compile(r"^[A-Za-z0-9._-]{2,64}@[A-Za-z][A-Za-z0-9]{1,32}$")
MOBILE_RE = re.compile(r"^[6-9]\d{9}$")

def now(): return datetime.now(timezone.utc)
def iso(dt): return dt.isoformat(timespec="seconds")
def valid_vpa(v): return bool(VPA_RE.match(v or ""))
def valid_mobile(m): return bool(MOBILE_RE.match(m or ""))

def pseudonym(value: str, domain: str) -> str:
    """Keyed, domain-separated pseudonym. Same input -> same id, but not reversible without the pepper."""
    return hmac.new(PEPPER.encode(), f"{domain}:{value.strip().lower()}".encode(), hashlib.sha256).hexdigest()[:24]

def mask_vpa(v: str) -> str:
    name, _, handle = v.partition("@")
    return f"{name[:1]}***@{handle}"

def notice():
    """Plain-English notice. DPDP Rules expect an itemised, specific, understandable notice with
    easy ways to withdraw consent, exercise rights and complain."""
    return {
        "version": NOTICE_VERSION,
        "what_we_use": DATA_TYPES,
        "period": f"the last {FI_DAYS} days",
        "why": "To suggest government schemes, insurance and savings options that may suit you. "
               "We do not use your data for anything else.",
        "how_you_share": "You approve the sharing in your Account Aggregator app. Your bank sends the "
                         "data to us in encrypted form. We never see your UPI PIN or bank password.",
        "how_long": f"We keep the result of our check for {DATA_LIFE_DAYS} days, then delete it. "
                    "We do not keep your raw transactions.",
        "sharing": "We will not give your data to any lender unless you agree again, separately.",
        "automated": "A computer program prepares your suggestion. You can ask a person to review it.",
        "your_rights": ["see your data", "correct it", "delete it", "withdraw consent at any time "
                        "(as easy as giving it)", "complain to us, and then to the Data Protection Board"],
        "grievance_contact": "grievance@example.org  (replace with your Data Protection Officer details)",
    }

SCHEMA = """
CREATE TABLE IF NOT EXISTS consents(id TEXT PRIMARY KEY, owner TEXT, subject TEXT, purpose TEXT, data_types TEXT,
  fi_days INT, data_life_days INT, notice_version TEXT, aa_handle TEXT, status TEXT,
  granted_at TEXT, expires_at TEXT, revoked_at TEXT, language TEXT);
CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY, owner TEXT, subject TEXT, consent_id TEXT,
  created_at TEXT, expires_at TEXT, status TEXT, masked_vpa TEXT, model_version TEXT, payload TEXT, review TEXT);
CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, role TEXT,
  action TEXT, resource TEXT, outcome TEXT, detail TEXT, prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS corrections(id TEXT PRIMARY KEY, owner TEXT, decision_id TEXT, field TEXT, old_value TEXT, new_value TEXT,
  reason TEXT, status TEXT, created_at TEXT, resolved_at TEXT, resolver TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS incidents(id TEXT PRIMARY KEY, declared_at TEXT, status TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS notifications(id TEXT PRIMARY KEY, owner TEXT, incident_id TEXT, created_at TEXT, read_at TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS flags(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS grievances(id TEXT PRIMARY KEY, owner TEXT, created_at TEXT, text TEXT,
  status TEXT, resolution TEXT);
"""

class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock: self.db.executescript(SCHEMA)

    # ---- audit (hash-chained) ------------------------------------------------
    def audit(self, actor, role, action, resource="", outcome="ok", **detail):
        with self.lock:
            row = self.db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else "GENESIS"
            entry = {"ts": iso(now()), "actor": actor, "role": role, "action": action,
                     "resource": resource, "outcome": outcome, "detail": json.dumps(detail, sort_keys=True, default=str)}
            h = hashlib.sha256((prev + json.dumps(entry, sort_keys=True)).encode()).hexdigest()
            self.db.execute("INSERT INTO audit(ts,actor,role,action,resource,outcome,detail,prev_hash,hash) "
                            "VALUES(?,?,?,?,?,?,?,?,?)",
                            (*entry.values(), prev, h))
            self.db.commit()

    def audit_rows(self, limit=200):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,))]

    def verify_audit(self):
        with self.lock:
            prev = "GENESIS"
            for r in self.db.execute("SELECT * FROM audit ORDER BY seq"):
                entry = {k: r[k] for k in ("ts", "actor", "role", "action", "resource", "outcome", "detail")}
                if r["prev_hash"] != prev or r["hash"] != hashlib.sha256((prev + json.dumps(entry, sort_keys=True)).encode()).hexdigest():
                    return False, r["seq"]
                prev = r["hash"]
            return True, None

    # ---- consent ledger --------------------------------------------------------
    def grant_consent(self, owner, subject, aa_handle, language="en"):
        cid, t = "c_" + uuid.uuid4().hex[:12], now()
        with self.lock:
            self.db.execute("INSERT INTO consents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, owner, subject, PURPOSE, json.dumps(DATA_TYPES), FI_DAYS, DATA_LIFE_DAYS, NOTICE_VERSION,
                 aa_handle, "ACTIVE", iso(t), iso(t + timedelta(days=DATA_LIFE_DAYS)), None, language))
            self.db.commit()
        return cid

    def get_consent(self, cid):
        with self.lock:
            r = self.db.execute("SELECT * FROM consents WHERE id=?", (cid,)).fetchone()
            return dict(r) if r else None

    def consent_is_active(self, cid, owner):
        c = self.get_consent(cid)
        return bool(c and c["owner"] == owner and c["status"] == "ACTIVE" and c["expires_at"] > iso(now()))

    def revoke_consent(self, cid, owner):
        """Withdrawal stops processing AND deletes the derived results that rested on that consent."""
        with self.lock:
            c = self.get_consent(cid)
            if not c or c["owner"] != owner: return None
            self.db.execute("UPDATE consents SET status='WITHDRAWN', revoked_at=? WHERE id=?", (iso(now()), cid))
            n = self.db.execute("DELETE FROM decisions WHERE consent_id=?", (cid,)).rowcount
            self.db.commit()
            return {"aa_handle": c["aa_handle"], "decisions_deleted": n}

    def consent_by_handle(self, handle):
        with self.lock:
            r = self.db.execute("SELECT * FROM consents WHERE aa_handle=?", (handle,)).fetchone()
            return dict(r) if r else None

    def list_consents(self):
        with self.lock:
            return [{k: r[k] for k in ("id", "subject", "purpose", "status", "granted_at", "expires_at", "revoked_at", "notice_version", "language")}
                    for r in self.db.execute("SELECT * FROM consents ORDER BY granted_at DESC")]

    # ---- decisions ---------------------------------------------------------------
    def save_decision(self, owner, subject, consent_id, masked_vpa, model_version, status, payload):
        did, t = "d_" + uuid.uuid4().hex[:12], now()
        with self.lock:
            self.db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (did, owner, subject, consent_id, iso(t), iso(t + timedelta(days=DATA_LIFE_DAYS)), status,
                 masked_vpa, model_version, json.dumps(payload, default=str), None))
            self.db.commit()
        return did

    def _dec(self, r):
        d = dict(r); d["payload"] = json.loads(d["payload"]); d["review"] = json.loads(d["review"]) if d["review"] else None
        return d

    def get_decision(self, did):
        with self.lock:
            r = self.db.execute("SELECT * FROM decisions WHERE id=?", (did,)).fetchone()
            return self._dec(r) if r else None

    def list_decisions(self, owner=None, status=None):
        q, a = "SELECT * FROM decisions WHERE 1=1", []
        if owner: q += " AND owner=?"; a.append(owner)
        if status: q += " AND status=?"; a.append(status)
        with self.lock:
            return [self._dec(r) for r in self.db.execute(q + " ORDER BY created_at DESC", a)]

    def update_decision(self, did, status=None, review=None, payload=None):
        with self.lock:
            if status: self.db.execute("UPDATE decisions SET status=? WHERE id=?", (status, did))
            if review is not None: self.db.execute("UPDATE decisions SET review=? WHERE id=?", (json.dumps(review), did))
            if payload is not None: self.db.execute("UPDATE decisions SET payload=? WHERE id=?", (json.dumps(payload, default=str), did))
            self.db.commit()

    # ---- data-subject rights + retention ---------------------------------------
    def export_owner(self, owner):
        with self.lock:
            return {"decisions": [{k: d[k] for k in ("id", "created_at", "expires_at", "status", "masked_vpa", "payload", "review")}
                                  for d in self.list_decisions(owner=owner)],
                    "consents": [dict(r) for r in self.db.execute("SELECT * FROM consents WHERE owner=?", (owner,))],
                    "grievances": [dict(r) for r in self.db.execute("SELECT * FROM grievances WHERE owner=?", (owner,))],
                    "corrections": [dict(r) for r in self.db.execute("SELECT * FROM corrections WHERE owner=?", (owner,))],
                    "notifications": [{**dict(r), "body": json.loads(r["body"])} for r in self.db.execute("SELECT * FROM notifications WHERE owner=?", (owner,))]}

    def erase_owner(self, owner):
        """Erase personal data. Audit rows stay (pseudonymous, legal-obligation retention) - state this in your notice."""
        with self.lock:
            n = self.db.execute("DELETE FROM decisions WHERE owner=?", (owner,)).rowcount
            self.db.execute("UPDATE consents SET status='WITHDRAWN', revoked_at=? WHERE owner=? AND status='ACTIVE'", (iso(now()), owner))
            g = self.db.execute("DELETE FROM grievances WHERE owner=?", (owner,)).rowcount
            k = self.db.execute("DELETE FROM corrections WHERE owner=?", (owner,)).rowcount
            self.db.execute("DELETE FROM notifications WHERE owner=?", (owner,))
            self.db.commit()
            return {"decisions_deleted": n, "grievances_deleted": g, "corrections_deleted": k}

    def purge_expired(self):
        with self.lock:
            t = iso(now())
            n = self.db.execute("DELETE FROM decisions WHERE expires_at < ?", (t,)).rowcount
            self.db.execute("UPDATE consents SET status='EXPIRED' WHERE status='ACTIVE' AND expires_at < ?", (t,))
            self.db.commit()
            return n

    # ---- grievances -------------------------------------------------------------
    def add_grievance(self, owner, text):
        gid = "g_" + uuid.uuid4().hex[:10]
        with self.lock:
            self.db.execute("INSERT INTO grievances VALUES(?,?,?,?,?,?)", (gid, owner, iso(now()), text[:2000], "OPEN", None))
            self.db.commit()
        return gid

    def list_grievances(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT id,created_at,text,status,resolution FROM grievances ORDER BY created_at DESC")]

    def resolve_grievance(self, gid, resolution):
        with self.lock:
            n = self.db.execute("UPDATE grievances SET status='RESOLVED', resolution=? WHERE id=?", (resolution[:2000], gid)).rowcount
            self.db.commit()
            return bool(n)

    # ---- corrections (right to correct / update) ------------------------------
    def add_correction(self, owner, did, field, old, new, reason):
        cid = "k_" + uuid.uuid4().hex[:10]
        with self.lock:
            self.db.execute("INSERT INTO corrections VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, owner, did, field, old, new, reason[:500], "OPEN", iso(now()), None, None, None))
            self.db.commit()
        return cid

    def get_correction(self, cid):
        with self.lock:
            r = self.db.execute("SELECT * FROM corrections WHERE id=?", (cid,)).fetchone()
            return dict(r) if r else None

    def list_corrections(self, owner=None, status=None):
        q, a = "SELECT * FROM corrections WHERE 1=1", []
        if owner: q += " AND owner=?"; a.append(owner)
        if status: q += " AND status=?"; a.append(status)
        with self.lock:
            return [dict(r) for r in self.db.execute(q + " ORDER BY created_at DESC", a)]

    def resolve_correction(self, cid, status, resolver, note):
        with self.lock:
            n = self.db.execute("UPDATE corrections SET status=?, resolved_at=?, resolver=?, note=? WHERE id=? AND status='OPEN'",
                                (status, iso(now()), resolver, note[:500], cid)).rowcount
            self.db.commit()
            return bool(n)

    # ---- flags (kill switch) -----------------------------------------------------
    def set_flag(self, k, v):
        with self.lock:
            self.db.execute("INSERT INTO flags VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))
            self.db.commit()

    def get_flag(self, k, default=None):
        with self.lock:
            r = self.db.execute("SELECT v FROM flags WHERE k=?", (k,)).fetchone()
            return json.loads(r["v"]) if r else default

    # ---- incidents + notifications -------------------------------------------------
    def save_incident(self, inc):
        with self.lock:
            self.db.execute("INSERT INTO incidents VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, data=excluded.data",
                            (inc["id"], inc["declared_at"], inc["status"], json.dumps(inc, default=str)))
            self.db.commit()

    def get_incident(self, iid):
        with self.lock:
            r = self.db.execute("SELECT data FROM incidents WHERE id=?", (iid,)).fetchone()
            return json.loads(r["data"]) if r else None

    def list_incidents(self):
        with self.lock:
            return [json.loads(r["data"]) for r in self.db.execute("SELECT data FROM incidents ORDER BY declared_at DESC")]

    def owners_in_scope(self, scope):
        """(owner, notice language) pairs for people whose data may be affected. Erased people are, by design, not listed."""
        with self.lock:
            if scope.get("all_active"):
                rows = self.db.execute("SELECT owner, language FROM consents WHERE status='ACTIVE' GROUP BY owner")
            else:
                rows = self.db.execute("SELECT c.owner AS owner, c.language AS language FROM consents c JOIN decisions d ON d.owner=c.owner "
                                       "WHERE d.created_at BETWEEN ? AND ? GROUP BY c.owner", (scope.get("from", ""), scope.get("to", "9999")))
            return [(r["owner"], r["language"] or "en") for r in rows]

    def add_notification(self, owner, incident_id, body):
        nid = "n_" + uuid.uuid4().hex[:10]
        with self.lock:
            self.db.execute("INSERT INTO notifications VALUES(?,?,?,?,?,?)", (nid, owner, incident_id, iso(now()), None, json.dumps(body, default=str)))
            self.db.commit()
        return nid

    def has_notification(self, owner, incident_id):
        with self.lock:
            return bool(self.db.execute("SELECT 1 FROM notifications WHERE owner=? AND incident_id=?", (owner, incident_id)).fetchone())

    def list_notifications(self, owner):
        with self.lock:
            return [{**dict(r), "body": json.loads(r["body"])} for r in self.db.execute("SELECT * FROM notifications WHERE owner=? ORDER BY created_at DESC", (owner,))]

    def mark_notification_read(self, nid, owner):
        with self.lock:
            n = self.db.execute("UPDATE notifications SET read_at=? WHERE id=? AND owner=? AND read_at IS NULL", (iso(now()), nid, owner)).rowcount
            self.db.commit()
            return bool(n)

    def notification_stats(self, incident_id):
        with self.lock:
            r = self.db.execute("SELECT COUNT(*) AS n, SUM(read_at IS NOT NULL) AS read FROM notifications WHERE incident_id=?", (incident_id,)).fetchone()
            return {"delivered_in_app": r["n"] or 0, "read": r["read"] or 0}
