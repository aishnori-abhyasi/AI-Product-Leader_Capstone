"""App <-> AA integration (simulator). Run: python test_aa_app.py"""
import json
from cryptography.hazmat.primitives.asymmetric import rsa
from aa_production import LocalX25519Crypto, ProductionAA, jws_sign_detached
from aa_simulator import make_simulator
from app import create_app
from test_aa import Session, BASE
from test_app import hdr, jpost, analyse
import rbac
from compliance import pseudonym

def main():
    fiu_k, aa_k, crypto, holder = rsa.generate_private_key(65537, 2048), rsa.generate_private_key(65537, 2048), LocalX25519Crypto(), {}
    sim_app, sim = make_simulator(crypto, fiu_k.public_key(), aa_k, lambda p, b, h: holder["c"].post(p, data=b, headers=h))
    aa = ProductionAA(BASE, "FIU-DEMO", "k", fiu_k, aa_k.public_key(), crypto, session=Session(sim_app.test_client()), fi_poll=(0, 3))
    app = create_app(env="sandbox", n_workers=400, aa=aa); c = app.test_client(); holder["c"] = c; store = app.store
    assert c.get("/healthz").get_json()["needs_aa_id"] is True
    tok = rbac.issue_token(pseudonym("9000000010", "mobile"), "customer")
    ok = lambda n: print("PASS", n)

    r = analyse(c, tok, "ravi@okaxis", aa_id="bad"); assert r.status_code == 400
    w = analyse(c, tok, "ravi@okaxis", aa_id="9876543200@sim"); assert w.status_code == 202; cid = w.get_json()["consent_id"]
    handle = store.get_consent(cid)["aa_handle"]
    assert jpost(c, f"/api/analyse/{cid}/continue", {}, tok).status_code == 202              # customer has not approved yet
    sim.approve(handle)                                                                      # customer taps approve; signed webhook hits the app
    r = jpost(c, f"/api/analyse/{cid}/continue", {}, tok); assert r.status_code == 200, r.get_json()
    res = r.get_json(); assert res["script"] and res["level"]; ok("live-AA flow: consent -> approval in AA app -> signed webhook -> analysis")
    assert "no_bill" not in json.dumps(res)
    print("      level:", res["level"], "| data_note:", res.get("data_note"))

    # forged webhook cannot withdraw / alter anything
    forged = json.dumps({"ConsentStatusNotification": {"consentHandle": handle, "consentStatus": "REVOKED"}}).encode()
    other = rsa.generate_private_key(65537, 2048)
    assert c.post("/api/aa/notify/consent", data=forged, headers={"x-jws-signature": jws_sign_detached(forged, other)}).status_code == 401
    assert c.post("/api/aa/notify/consent", data=forged).status_code == 401
    assert store.get_consent(cid)["status"] == "ACTIVE"; ok("forged / unsigned webhooks rejected, state unchanged")

    # customer withdraws consent inside the AA app -> we delete derived data
    assert len(c.get("/api/my/decisions", headers=hdr(tok)).get_json()) == 1
    sim.revoke_by_customer(handle)
    assert store.get_consent(cid)["status"] == "WITHDRAWN" and c.get("/api/my/decisions", headers=hdr(tok)).get_json() == []
    assert any(r["action"] == "consent_withdrawn_at_aa" for r in store.audit_rows(100)); ok("withdrawal in the AA app propagates: consent withdrawn, results deleted, audited")
    assert store.verify_audit()[0]; print("\nALL AA-APP TESTS PASSED")

if __name__ == "__main__": main()
