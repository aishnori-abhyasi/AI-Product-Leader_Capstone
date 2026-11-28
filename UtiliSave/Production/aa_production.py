"""
Account Aggregator adapter (FIU side), shaped after the ReBIT / Sahamati AA specs.

HONEST STATUS
  Built from public documentation and TESTED ONLY against aa_simulator.py, which encodes my reading of those
  docs. It has NOT been run against a real AA, so field names, versions, headers and the crypto details MUST be
  validated in Sahamati's UAT sandbox before any real use. Specific known risks:
    * Real FIPs exchange Curve25519 keys in a Weierstrass-style encoding (Rahasya). Do not use LocalX25519Crypto
      in production - run Sahamati's Rahasya service and use RahasyaCrypto (its request shapes below are from
      memory of its README: verify them).
    * Endpoint paths / `ver` differ between spec versions (1.x vs 2.x); ConsentDetail fields and purpose codes must
      match the Sahamati Fair Use consent template for your use case.
    * Train-serve skew: features derived here from REAL bank statements are PROXIES that differ from the synthetic
      feature definitions the model was trained on. Retrain on features produced by statement_to_telemetry().

DESIGN POINTS THAT ARE SPEC-INDEPENDENT AND ENFORCED HERE
  * data minimisation: DEPOSIT accounts, TRANSACTIONS only, last 90 days, ONETIME fetch, short DataLife
  * every outbound call carries a detached JWS (RFC 7797, b64=false) signature; inbound notifications are
    verified against the AA's public key and rejected otherwise
  * statements are decrypted and mapped in memory; nothing raw is persisted
"""
import base64, hashlib, json, os, re, time, uuid
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization

VER = "2.0.0"

def b64(b): return base64.b64encode(b).decode()
def b64d(s): return base64.b64decode(s)
def b64u(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def b64ud(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
def now_iso(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

# ----------------------------------------------------------------- detached JWS (RFC 7515 + 7797)
def jws_sign_detached(body: bytes, private_key, kid="fiu-key-1") -> str:
    h = b64u(json.dumps({"alg": "RS256", "kid": kid, "b64": False, "crit": ["b64"]}, separators=(",", ":")).encode())
    sig = private_key.sign(h.encode() + b"." + body, padding.PKCS1v15(), hashes.SHA256())
    return f"{h}..{b64u(sig)}"

def jws_verify_detached(body: bytes, value: str, public_key) -> bool:
    try:
        h, mid, s = value.split(".")
        if mid or json.loads(b64ud(h)).get("alg") != "RS256": return False
        public_key.verify(b64ud(s), h.encode() + b"." + body, padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False

# ----------------------------------------------------------------- crypto backends
class LocalX25519Crypto:
    """TEST ONLY. RFC 7748 X25519 + XOR(nonces) -> HKDF-SHA256 -> AES-256-GCM (salt = first 20 bytes, iv = last 12).
    Not interoperable with real FIPs. Used by the simulator so the whole flow can be exercised offline."""
    def generate(self):
        priv = X25519PrivateKey.generate(); nonce = os.urandom(32)
        raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        km = {"cryptoAlg": "ECDH", "curve": "Curve25519", "params": "AES/GCM/NoPadding",
              "DHPublicKey": {"expiry": (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                              "Parameters": "", "KeyValue": b64(raw)}, "Nonce": b64(nonce)}
        return priv.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()), km

    def _key_iv(self, our_priv, remote_km, our_km):
        shared = X25519PrivateKey.from_private_bytes(our_priv).exchange(X25519PublicKey.from_public_bytes(b64d(remote_km["DHPublicKey"]["KeyValue"])))
        x = bytes(a ^ b for a, b in zip(b64d(remote_km["Nonce"]), b64d(our_km["Nonce"])))
        return HKDF(hashes.SHA256(), 32, salt=x[:20], info=None).derive(shared), x[-12:]

    def decrypt(self, enc_b64, remote_km, our_priv, our_km):
        key, iv = self._key_iv(our_priv, remote_km, our_km)
        return AESGCM(key).decrypt(iv, b64d(enc_b64), None)

    def encrypt(self, plaintext, remote_km, our_priv, our_km):                 # FIP side (simulator)
        key, iv = self._key_iv(our_priv, remote_km, our_km)
        return b64(AESGCM(key).encrypt(iv, plaintext, None))

class RahasyaCrypto:
    """PRODUCTION: delegate key generation and decryption to a Rahasya service you run (Sahamati open source).
    Request/response shapes below are from memory - VERIFY against the Rahasya README before use."""
    def __init__(self, base_url, session, timeout=10): self.u, self.s, self.t = base_url.rstrip("/"), session, timeout
    def generate(self):
        r = self.s.request("GET", f"{self.u}/ecc/v1/generateKey", timeout=self.t); r.raise_for_status()
        j = r.json(); return j["privateKey"], j["KeyMaterial"]
    def decrypt(self, enc_b64, remote_km, our_priv, our_km):
        r = self.s.request("POST", f"{self.u}/ecc/v1/decrypt", timeout=self.t, json={
            "base64Data": enc_b64, "base64RemoteNonce": remote_km["Nonce"], "base64YourNonce": our_km["Nonce"],
            "ourPrivateKey": our_priv, "remoteKeyMaterial": remote_km})
        r.raise_for_status(); return b64d(r.json()["data"]) if "data" in r.json() else b64d(r.json()["base64Data"])

# ----------------------------------------------------------------- statements -> telemetry
ESSENTIAL = re.compile(r"grocer|kirana|supermarket|mart\b|medical|pharm|rent|electric|water|gas|bbps|broadband|recharge|dth|postpaid", re.I)
BILL = re.compile(r"bbps|electric|water board|gas|broadband|postpaid|dth|recharge", re.I)

def _find_transactions(o):
    """Tolerant search for the transaction list (schema/casing differ between FIPs and spec versions)."""
    if isinstance(o, list) and o and isinstance(o[0], dict) and {"amount"} <= {k.lower() for k in o[0]}: return o
    if isinstance(o, dict):
        for v in o.values():
            r = _find_transactions(v)
            if r: return r
    if isinstance(o, list):
        for v in o:
            r = _find_transactions(v)
            if r: return r
    return []

def _g(d, k, default=None):
    return next((v for kk, v in d.items() if kk.lower() == k), default)

def _counterparty(narr):
    parts = [p.strip() for p in str(narr).split("/")]
    key = parts[3] if len(parts) >= 5 and parts[0].upper() == "UPI" else str(narr)       # bank-specific: validate per FIP
    return re.sub(r"\d+", "", key).lower().strip() or "unknown"

def statement_to_telemetry(txns, subject, window_end: date, days=90) -> pd.DataFrame:
    start = window_end - timedelta(days=days - 1)
    inflow, ess, bal_known = np.zeros(days), np.zeros(days), {}
    payers, bill_blocks = set(), set()
    for t in sorted(txns, key=lambda x: str(_g(x, "transactiontimestamp") or _g(x, "valuedate") or "")):
        try: d = (datetime.fromisoformat(str(_g(t, "valuedate") or _g(t, "transactiontimestamp"))[:10]).date() - start).days
        except ValueError: continue
        if not 0 <= d < days: continue
        amt, typ, narr = float(_g(t, "amount", 0)), str(_g(t, "type", "")).upper(), str(_g(t, "narration", ""))
        if _g(t, "currentbalance") is not None: bal_known[d] = float(_g(t, "currentbalance"))
        if typ == "CREDIT" and str(_g(t, "mode", "UPI")).upper() == "UPI":
            inflow[d] += amt; payers.add(_counterparty(narr))
        elif typ == "DEBIT" and ESSENTIAL.search(narr):
            ess[d] += amt
            if BILL.search(narr): bill_blocks.add(d // 30)
    bal, last = np.zeros(days), (next(iter(bal_known.values()), 0.0) if bal_known else 0.0)
    for d in range(days):
        last = bal_known.get(d, last); bal[d] = last
    # PROXY for "bills on time": regularity of recurring-bill payments per 30-day block (no due dates in a statement)
    nblocks = (days + 29) // 30
    recurring = len(bill_blocks) >= 2
    due = np.zeros(days, bool); ontime = np.zeros(days, bool)
    if recurring:
        for b in range(nblocks):
            due[min(b * 30, days - 1)] = True; ontime[min(b * 30, days - 1)] = b in bill_blocks
    ids = {p: i for i, p in enumerate(sorted(payers))}
    tel = pd.DataFrame({"worker_id": subject, "day": np.arange(days), "date": pd.date_range(start, periods=days),
                        "inflow": inflow, "payer_id": np.where(inflow > 0, 0, -1), "essential_out": ess,
                        "closing_balance": bal, "bill_due": due, "bill_on_time": ontime})
    tel.attrs["feature_overrides"] = {"payer_diversity": float(len(ids))}
    tel.attrs["no_bill_signal"] = not recurring           # absence of a signal is NOT evidence of late payment
    return tel

# ----------------------------------------------------------------- the adapter
class ProductionAA:
    def __init__(self, base_url, fiu_id, api_key, signing_key, aa_public_key, crypto, session=None,
                 purpose_code="102", timeout=15, fi_poll=(0.2, 20)):
        import requests
        self.u, self.fiu, self.key, self.sk, self.aa_pk, self.crypto = base_url.rstrip("/"), fiu_id, api_key, signing_key, aa_public_key, crypto
        self.s, self.purpose, self.t, self.poll = session or requests.Session(), purpose_code, timeout, fi_poll
        self._c = {}

    def _call(self, method, path, body=None):
        raw = json.dumps(body, separators=(",", ":")).encode() if body is not None else b""
        h = {"Content-Type": "application/json", "client_api_key": self.key, "x-jws-signature": jws_sign_detached(raw, self.sk)}
        r = self.s.request(method, self.u + path, data=raw or None, headers=h, timeout=self.t)
        return r

    # --- consent
    def request_consent(self, subject, purpose, fi_days, data_life_days, hint="", aa_id=None):
        if not aa_id or not re.fullmatch(r"\d{10}@[A-Za-z0-9._-]{2,40}", aa_id): raise ValueError("a valid AA handle (mobile@provider) is required")
        t, end = datetime.now(timezone.utc), datetime.now(timezone.utc)
        fmt = lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        detail = {"consentStart": fmt(t), "consentExpiry": fmt(t + timedelta(days=1)), "consentMode": "VIEW", "fetchType": "ONETIME",
                  "consentTypes": ["TRANSACTIONS"], "fiTypes": ["DEPOSIT"],                       # minimisation: nothing else
                  "DataConsumer": {"id": self.fiu, "type": "FIU"}, "Customer": {"id": aa_id},
                  "Purpose": {"code": self.purpose, "text": "Suggest government schemes and savings options", "Category": {"type": "string"}},
                  "FIDataRange": {"from": fmt(t - timedelta(days=fi_days)), "to": fmt(end)},
                  "DataLife": {"unit": "DAY", "value": data_life_days}}
        r = self._call("POST", "/Consent", {"ver": VER, "timestamp": now_iso(), "txnid": str(uuid.uuid4()), "ConsentDetail": detail})
        if r.status_code != 200: raise RuntimeError(f"AA rejected consent request: {r.status_code}")
        h = r.json()["ConsentHandle"]
        self._c[h] = {"subject": subject, "status": "PENDING", "consent_id": None, "fi_days": fi_days}
        return h                                         # redirect the customer to their AA app to approve

    def handle_notification(self, payload):
        n = payload.get("ConsentStatusNotification", {}); h = n.get("consentHandle")
        if h in self._c:
            st = {"ACTIVE": "APPROVED", "REJECTED": "REJECTED", "REVOKED": "REVOKED", "PAUSED": "REVOKED", "EXPIRED": "REVOKED"}.get(n.get("consentStatus"))
            if st: self._c[h]["status"] = st; self._c[h]["consent_id"] = n.get("consentId") or self._c[h]["consent_id"]
            return True
        return False

    def verify_notification(self, body: bytes, header: str): return jws_verify_detached(body, header or "", self.aa_pk)

    def status(self, handle):
        c = self._c.get(handle)
        if not c: return "UNKNOWN"
        if c["status"] == "PENDING":                      # webhook may be delayed: ask the AA directly
            r = self._call("GET", f"/Consent/handle/{handle}")
            if r.status_code == 200:
                s = r.json().get("ConsentStatus", {})
                self.handle_notification({"ConsentStatusNotification": {"consentHandle": handle, "consentId": s.get("id"), "consentStatus": s.get("status")}})
        return self._c[handle]["status"]

    def revoke(self, handle):
        c = self._c.get(handle)
        if c and c["consent_id"]: self._call("POST", "/Consent/revoke", {"ver": VER, "consentId": c["consent_id"]})
        if c: c["status"] = "REVOKED"

    # --- data
    def fetch(self, handle):
        c = self._c.get(handle)
        if not c or self.status(handle) != "APPROVED": raise PermissionError("consent not approved")
        sc = self._call("GET", f"/Consent/{c['consent_id']}"); sc.raise_for_status()
        sig = sc.json()["signedConsent"].split(".")[2]
        priv, km = self.crypto.generate()
        to = datetime.now(timezone.utc); fr = to - timedelta(days=c["fi_days"]); fmt = lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rq = self._call("POST", "/FI/request", {"ver": VER, "timestamp": now_iso(), "txnid": str(uuid.uuid4()),
                        "FIDataRange": {"from": fmt(fr), "to": fmt(to)}, "Consent": {"id": c["consent_id"], "digitalSignature": sig}, "KeyMaterial": km})
        if rq.status_code != 200: raise RuntimeError(f"FI request failed: {rq.status_code}")
        sid = rq.json()["sessionId"]
        for _ in range(self.poll[1]):
            r = self._call("GET", f"/FI/fetch/{sid}")
            if r.status_code == 200: break
            time.sleep(self.poll[0])
        else:
            raise TimeoutError("financial data not ready")
        txns = []
        for fip in r.json().get("FI", []):
            for acct in fip.get("data", []):
                body = json.loads(self.crypto.decrypt(acct["encryptedFI"], fip["KeyMaterial"], priv, km))
                txns += _find_transactions(body)
        del priv
        tel = statement_to_telemetry(txns, c["subject"], to.date(), c["fi_days"])
        c["status"] = "FETCHED"                          # ONETIME consent: do not fetch again
        return tel
