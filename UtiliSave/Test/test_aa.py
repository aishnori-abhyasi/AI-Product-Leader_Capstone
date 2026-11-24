"""AA adapter tests against the simulator. Run: python test_aa.py"""
import json
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidTag

from aa_production import (LocalX25519Crypto, ProductionAA, jws_sign_detached, jws_verify_detached, statement_to_telemetry, b64, b64d)
from aa_simulator import make_simulator
from upi_agents_mvp import engineer_features

BASE = "http://aa.sim"
class Resp:
    def __init__(s, r): s.status_code, s._r = r.status_code, r
    def json(s): return s._r.get_json()
    def raise_for_status(s):
        if s.status_code >= 400: raise RuntimeError(s.status_code)
class Session:
    def __init__(s, client): s.c = client
    def request(s, method, url, data=None, headers=None, timeout=None, json=None):
        return Resp(s.c.open(url.replace(BASE, ""), method=method, data=data, headers=headers or {}))

def build():
    rsa_k = lambda: rsa.generate_private_key(65537, 2048)
    fiu_k, aa_k, crypto = rsa_k(), rsa_k(), LocalX25519Crypto()
    holder = {}
    def notify(path, body, headers):
        ok = holder["aa"].verify_notification(body, headers.get("x-jws-signature"))
        if ok: holder["aa"].handle_notification(json.loads(body))
        return ok
    sim_app, sim = make_simulator(crypto, fiu_k.public_key(), aa_k, notify)
    aa = ProductionAA(BASE, "FIU-DEMO", "key", fiu_k, aa_k.public_key(), crypto, session=Session(sim_app.test_client()), fi_poll=(0, 3))
    holder["aa"] = aa
    return aa, sim, sim_app, aa_k

def main():
    ok = lambda n: print("PASS", n)
    aa, sim, sim_app, aa_k = build()

    # 1 JWS
    body = b'{"a":1}'; k2 = rsa.generate_private_key(65537, 2048)
    s = jws_sign_detached(body, aa_k)
    assert jws_verify_detached(body, s, aa_k.public_key()) and not jws_verify_detached(body + b" ", s, aa_k.public_key())
    assert not jws_verify_detached(body, s, k2.public_key()) and not jws_verify_detached(body, "", aa_k.public_key()); ok("detached JWS: valid / tampered / wrong key / missing")

    # 2 crypto
    c = LocalX25519Crypto(); fiu_p, fiu_km = c.generate(); fip_p, fip_km = c.generate()
    enc = c.encrypt(b"secret statement", fiu_km, fip_p, fip_km)
    assert c.decrypt(enc, fip_km, fiu_p, fiu_km) == b"secret statement"
    bad = bytearray(b64d(enc)); bad[3] ^= 1
    try: c.decrypt(b64(bytes(bad)), fip_km, fiu_p, fiu_km); raise SystemExit("tamper undetected")
    except InvalidTag: pass; ok("key exchange + AES-GCM round-trip; tampering detected")

    # 3 request minimisation + validation
    try: aa.request_consent("subj1", "p", 90, 30, aa_id="not-an-id"); raise SystemExit("bad id accepted")
    except ValueError: pass
    h = aa.request_consent("subj1", "p", 90, 30, aa_id="9876543201@sim")
    d = sim.state["consents"][h]["detail"] if hasattr(sim, "state") else None
    assert d["fiTypes"] == ["DEPOSIT"] and d["consentTypes"] == ["TRANSACTIONS"] and d["fetchType"] == "ONETIME"
    assert d["DataLife"] == {"unit": "DAY", "value": 30} and d["Customer"]["id"] == "9876543201@sim"
    span = (datetime.fromisoformat(d["FIDataRange"]["to"][:10]) - datetime.fromisoformat(d["FIDataRange"]["from"][:10])).days
    assert span == 90; ok("consent request is minimal: DEPOSIT, TRANSACTIONS only, ONETIME, 90 days, 30-day data life")

    # 4 flow: pending -> approve (signed webhook) -> fetch -> features
    assert aa.status(h) == "PENDING"
    sim.approve(h); assert aa.status(h) == "APPROVED"
    tel = aa.fetch(h)
    assert len(tel) == 90 and {"inflow", "payer_id", "essential_out", "closing_balance", "bill_due", "bill_on_time"} <= set(tel.columns)
    F = engineer_features(tel); F.loc[:, "payer_diversity"] = tel.attrs["feature_overrides"]["payer_diversity"]
    print("      steady-id features:", {k: round(float(v), 2) for k, v in F.iloc[0].items()}, "| no_bill_signal:", tel.attrs["no_bill_signal"])
    ok("full flow through simulator: consent -> signed notification -> encrypted fetch -> telemetry -> features")

    # 5 ONETIME
    try: aa.fetch(h); raise SystemExit("second fetch allowed")
    except PermissionError: ok("ONETIME consent cannot be fetched twice")

    # 6 archetype separation survives the statement round-trip (mapper sanity, not real-world validity)
    def feats(aa_id):
        hh = aa.request_consent("s-" + aa_id, "p", 90, 30, aa_id=aa_id); sim.approve(hh); t = aa.fetch(hh)
        f = engineer_features(t); return f.iloc[0]
    st, ir = feats("9876543200@sim"), feats("9876543209@sim")
    assert st["active_days_ratio"] > ir["active_days_ratio"] and st["inflow_cv"] < ir["inflow_cv"], (st.to_dict(), ir.to_dict())
    ok(f"mapper keeps signal: active days {st['active_days_ratio']:.2f} (steady) vs {ir['active_days_ratio']:.2f} (irregular)")

    # 7 forged / unsigned webhook + simulator rejects unsigned FIU calls
    forged = json.dumps({"ConsentStatusNotification": {"consentHandle": h, "consentStatus": "ACTIVE"}}).encode()
    other = rsa.generate_private_key(65537, 2048)
    assert not aa.verify_notification(forged, jws_sign_detached(forged, other)) and not aa.verify_notification(forged, None)
    r = sim_app.test_client().post("/Consent", data=b"{}", headers={}); assert r.status_code == 401; ok("forged/unsigned notification rejected; unsigned FIU call rejected")

    # 8 customer revokes in AA app -> status follows, fetch refused
    h2 = aa.request_consent("s2", "p", 90, 30, aa_id="9876543202@sim"); sim.approve(h2); sim.revoke_by_customer(h2)
    assert aa.status(h2) == "REVOKED"
    try: aa.fetch(h2); raise SystemExit("fetched after revoke")
    except PermissionError: ok("revocation in the AA app propagates; fetch refused")

    # 9 mapper edge cases: no bills => flagged as missing signal, empty statement safe
    t0 = statement_to_telemetry([], "x", datetime.now().date()); assert t0.attrs["no_bill_signal"] and t0["inflow"].sum() == 0
    tx = [{"type": "CREDIT", "mode": "UPI", "amount": "500", "valueDate": str(datetime.now().date()), "narration": "UPI/CR/1/RAVI/BANK/ravi@upi", "currentBalance": "900"}]
    t1 = statement_to_telemetry(tx, "x", datetime.now().date()); assert t1["inflow"].sum() == 500 and t1.attrs["no_bill_signal"]; ok("mapper: missing bill signal is flagged, not treated as late payment")
    print("\nALL AA TESTS PASSED")

if __name__ == "__main__": main()
