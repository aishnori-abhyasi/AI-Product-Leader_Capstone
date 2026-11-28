"""
Account Aggregator + bank (FIP) simulator for tests and demos. Encodes MY reading of the AA contract; it is not the
real thing. It: verifies the FIU's detached JWS, enforces ONETIME fetch and consent status, signs notifications with
the AA key, and returns an ENCRYPTED synthetic DEPOSIT statement built from the telemetry generator.
"""
import hashlib, json, uuid
from datetime import datetime, timedelta, timezone

import numpy as np
from flask import Flask, jsonify, request

from aa_production import b64, jws_sign_detached, jws_verify_detached, now_iso
from upi_agents_mvp import generate_upi_telemetry

def telemetry_to_deposit_json(tel, end_date, seed):
    """Inverse of statement_to_telemetry: day rows -> bank transactions with realistic UPI narrations."""
    r = np.random.default_rng(seed); txns = []; start = end_date - timedelta(days=len(tel) - 1)
    for _, row in tel.iterrows():
        day = start + timedelta(days=int(row["day"])); k = 0
        if row["inflow"] > 0:
            parts = r.dirichlet(np.ones(r.integers(1, 4))) * row["inflow"]
            for amt in parts:
                pid = int(r.integers(0, 12)); k += 1
                txns.append({"type": "CREDIT", "mode": "UPI", "amount": f"{amt:.2f}", "valueDate": str(day), "txnId": uuid.uuid4().hex[:10],
                             "narration": f"UPI/CR/{r.integers(10**11, 10**12)}/PAYER{chr(65 + pid)}/BANKX/payer{pid}@upi",
                             "currentBalance": f"{row['closing_balance']:.2f}"})
        if row["essential_out"] > 0:
            txns.append({"type": "DEBIT", "mode": "UPI", "amount": f"{row['essential_out']:.2f}", "valueDate": str(day), "txnId": uuid.uuid4().hex[:10],
                         "narration": f"UPI/DR/{r.integers(10**11, 10**12)}/KIRANA GROCERY/BANKY/kirana@upi", "currentBalance": f"{row['closing_balance']:.2f}"})
        if row["bill_due"] and row["bill_on_time"]:
            txns.append({"type": "DEBIT", "mode": "OTHERS", "amount": "350.00", "valueDate": str(day), "txnId": uuid.uuid4().hex[:10],
                         "narration": "BBPS ELECTRICITY BOARD", "currentBalance": f"{row['closing_balance']:.2f}"})
    return {"account": {"type": "DEPOSIT", "maskedAccNumber": "XXXXXX1234", "transactions": {"transaction": txns}}}

def make_simulator(crypto, fiu_public_key, aa_private_key, notify_fiu):
    """notify_fiu(path, body_bytes, headers) delivers the AA's webhook to the FIU."""
    app, S = Flask("aa-sim"), {"consents": {}, "sessions": {}}

    def guard():
        if not jws_verify_detached(request.get_data(), request.headers.get("x-jws-signature", ""), fiu_public_key):
            return jsonify(errorCode="InvalidSignature"), 401

    @app.post("/Consent")
    def consent():
        bad = guard()
        if bad: return bad
        d = request.get_json()["ConsentDetail"]
        h = "h-" + uuid.uuid4().hex[:10]
        S["consents"][h] = {"detail": d, "status": "PENDING", "id": None, "uses": 0}
        return jsonify(ver="2.0.0", txnid=request.get_json()["txnid"], ConsentHandle=h)

    @app.get("/Consent/handle/<h>")
    def handle(h):
        c = S["consents"].get(h)
        if not c: return jsonify(errorCode="NotFound"), 404
        return jsonify(ConsentHandle=h, ConsentStatus={"id": c["id"], "status": c["status"]})

    @app.get("/Consent/<cid>")
    def signed(cid):
        c = next((x for x in S["consents"].values() if x["id"] == cid), None)
        if not c or c["status"] != "ACTIVE": return jsonify(errorCode="NotFound"), 404
        body = json.dumps(c["detail"], separators=(",", ":")).encode()
        return jsonify(signedConsent=jws_sign_detached(body, aa_private_key, "aa-key-1").replace("..", "." + "e30" + "."))

    @app.post("/Consent/revoke")
    def revoke():
        c = next((x for x in S["consents"].values() if x["id"] == request.get_json().get("consentId")), None)
        if c: c["status"] = "REVOKED"
        return jsonify(ok=True)

    @app.post("/FI/request")
    def fi_request():
        bad = guard()
        if bad: return bad
        j = request.get_json(); c = next((x for x in S["consents"].values() if x["id"] == j["Consent"]["id"]), None)
        if not c or c["status"] != "ACTIVE": return jsonify(errorCode="InvalidConsent"), 403
        if c["detail"]["fetchType"] == "ONETIME" and c["uses"] >= 1: return jsonify(errorCode="ConsentUsed"), 403
        if not j["KeyMaterial"].get("Nonce"): return jsonify(errorCode="BadKeyMaterial"), 400
        c["uses"] += 1
        cust = c["detail"]["Customer"]["id"]; digit = int(cust[9])                         # sim-only: last digit picks the archetype
        arch = "steady" if digit <= 3 else "mixed" if digit <= 6 else "irregular"
        seed = int(hashlib.sha256(cust.encode()).hexdigest()[:8], 16)
        days = (datetime.fromisoformat(j["FIDataRange"]["to"][:10]) - datetime.fromisoformat(j["FIDataRange"]["from"][:10])).days
        tel, _ = generate_upi_telemetry(n_workers=1, days=days, seed=seed, force_archetype=arch)
        stmt = telemetry_to_deposit_json(tel, datetime.now(timezone.utc).date(), seed)
        fip_priv, fip_km = crypto.generate()
        enc = crypto.encrypt(json.dumps(stmt).encode(), j["KeyMaterial"], fip_priv, fip_km)
        sid = "s-" + uuid.uuid4().hex[:10]
        S["sessions"][sid] = {"FI": [{"fipID": "SIM-BANK", "KeyMaterial": fip_km, "data": [{"linkRefNumber": "L1", "maskedAccNumber": "XXXXXX1234", "encryptedFI": enc}]}]}
        return jsonify(ver="2.0.0", sessionId=sid, consentId=c["id"])

    @app.get("/FI/fetch/<sid>")
    def fi_fetch(sid):
        bad = guard()
        if bad: return bad
        s = S["sessions"].get(sid)
        return (jsonify(ver="2.0.0", **s) if s else (jsonify(errorCode="NotFound"), 404))

    class Sim:                                              # test/demo controls = what the customer does in the AA app
        def approve(self, handle_id):
            c = S["consents"][handle_id]; c["status"], c["id"] = "ACTIVE", "c-" + uuid.uuid4().hex[:10]; self._notify(handle_id)
        def reject(self, handle_id): S["consents"][handle_id]["status"] = "REJECTED"; self._notify(handle_id)
        def revoke_by_customer(self, handle_id): S["consents"][handle_id]["status"] = "REVOKED"; self._notify(handle_id)
        def _notify(self, handle_id):
            c = S["consents"][handle_id]
            body = json.dumps({"ver": "2.0.0", "timestamp": now_iso(), "txnid": str(uuid.uuid4()),
                               "Notifier": {"type": "AA", "id": "SIM-AA"},
                               "ConsentStatusNotification": {"consentId": c["id"], "consentHandle": handle_id, "consentStatus": c["status"]}},
                              separators=(",", ":")).encode()
            return notify_fiu("/api/aa/notify/consent", body, {"Content-Type": "application/json", "x-jws-signature": jws_sign_detached(body, aa_private_key, "aa-key-1")})
    sim = Sim(); sim.state = S
    return app, sim
