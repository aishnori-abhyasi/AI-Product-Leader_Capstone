"""Run:  python test_app.py     (also pytest-compatible)"""
import json, os, sqlite3

import numpy as np

import rbac
from app import create_app
from compliance import NOTICE_VERSION, pseudonym
from explain import _logit

VPA_A, VPA_B = "steady.ravi@okaxis", "irregular.meena@ybl"
MOB_A, MOB_B = "9876543210", "9123456780"
STAFF_PW = os.getenv("STAFF_DEMO_PASSWORD", "Demo-Staff-Pass-1")

def jpost(c, url, body=None, tok=None, method="post"):
    return getattr(c, method)(url, json=body if body is not None else {}, headers={"Authorization": f"Bearer {tok}"} if tok else {})

def hdr(tok): return {"Authorization": f"Bearer {tok}"}

def customer(c, mobile):
    otp = jpost(c, "/api/auth/otp/request", {"mobile": mobile}).get_json()["sandbox_otp"]
    r = jpost(c, "/api/auth/otp/verify", {"mobile": mobile, "otp": otp})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["token"]

def analyse(c, tok, vpa, occ="delivery rider", **over):
    body = {"vpa": vpa, "occupation": occ, "adult": True, "consent": True, "notice_version": NOTICE_VERSION, **over}
    return jpost(c, "/api/analyse", body, tok)

def store_handle(app, cid): return app.store.get_consent(cid)["aa_handle"]

def staff(c, user):
    r = jpost(c, "/api/staff/login", {"username": user, "password": STAFF_PW})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["token"]

def main():
    app = create_app(env="sandbox", n_workers=400)
    c, store = app.test_client(), app.store
    ok = lambda name: print("PASS", name)

    # 1 page, headers, notice
    r = c.get("/")
    assert r.status_code == 200 and "nonce-" in r.headers["Content-Security-Policy"] and "script-src 'none'" not in r.headers["Content-Security-Policy"]
    assert c.get("/api/notice").get_json()["version"] == NOTICE_VERSION; ok("page + CSP nonce + notice")

    # 2 auth required
    assert jpost(c, "/api/analyse", {}).status_code == 401
    assert c.get("/api/staff/cases").status_code == 401
    bad = rbac.issue_token("x", "customer") + "tamper"
    assert c.get("/api/my/decisions", headers=hdr(bad)).status_code == 401; ok("unauthenticated / tampered token rejected")

    # 3 consent gating
    ta, tb = customer(c, MOB_A), customer(c, MOB_B)
    assert analyse(c, ta, VPA_A, consent=False).status_code == 400
    assert analyse(c, ta, VPA_A, adult=False).status_code == 400
    assert analyse(c, ta, VPA_A, notice_version="old").status_code == 400
    assert analyse(c, ta, "not-a-vpa").status_code == 400
    assert not store.list_consents(); ok("no processing without notice, consent, adult confirmation")

    # 4 full flow + customer response never leaks model internals / raw ids
    r = analyse(c, ta, VPA_A); assert r.status_code == 200, r.get_json()
    ra = r.get_json(); txt = json.dumps(ra)
    assert ra["script"] and ra["next_steps"] and ra["why"] and ra["level"] in ("Getting started", "Building", "Growing", "Strong")
    assert not any(k in ra for k in ("p_repay", "shap", "shap_log_odds", "segment")) and VPA_A not in txt
    assert "automated" in ra["disclosure"]; ok("end-to-end suggestion in plain English, no internals leaked")
    rb = analyse(c, tb, VPA_B, occ="ride-app driver").get_json()

    # 5 IDOR
    assert c.get(f"/api/my/decisions/{ra['id']}", headers=hdr(tb)).status_code == 404
    assert jpost(c, f"/api/my/decisions/{ra['id']}/contest", {"reason": "x"}, tb).status_code == 404
    assert c.get(f"/api/my/decisions/{ra['id']}", headers=hdr(ta)).status_code == 200
    assert len(c.get("/api/my/decisions", headers=hdr(tb)).get_json()) == 1; ok("customer B cannot read/contest customer A's decision")

    # 6 RBAC matrix  (expected HTTP status per role per endpoint)
    toks = {u: staff(c, u) for u in ("support1", "analyst1", "dpo1", "admin1")}
    toks["customer"] = ta
    did = ra["id"]
    matrix = {  # endpoint -> {role: expected}
        ("get", "/api/staff/cases"):        dict(support1=200, analyst1=200, dpo1=403, admin1=403, customer=403),
        ("get", "/api/staff/model-report"): dict(support1=403, analyst1=200, dpo1=200, admin1=403, customer=403),
        ("get", "/api/staff/fairness"):     dict(support1=403, analyst1=200, dpo1=200, admin1=403, customer=403),
        ("get", "/api/staff/audit"):        dict(support1=403, analyst1=403, dpo1=200, admin1=403, customer=403),
        ("get", "/api/staff/consents"):     dict(support1=403, analyst1=403, dpo1=200, admin1=403, customer=403),
        ("get", "/api/staff/grievances"):   dict(support1=200, analyst1=403, dpo1=200, admin1=403, customer=403),
        ("get", "/api/staff/users"):        dict(support1=403, analyst1=403, dpo1=403, admin1=200, customer=403),
        ("post", "/api/staff/purge"):       dict(support1=403, analyst1=403, dpo1=200, admin1=403, customer=403),
        ("get", f"/api/staff/cases/{did}"): dict(support1=200, analyst1=200, dpo1=403, admin1=403, customer=403),
    }
    for (m, url), exp in matrix.items():
        for who, code in exp.items():
            got = getattr(c, m)(url, headers=hdr(toks[who])).status_code
            assert got == code, f"{m} {url} as {who}: expected {code}, got {got}"
    assert c.get("/api/my/decisions", headers=hdr(toks["support1"])).status_code == 403          # staff are not customers
    ok(f"RBAC matrix ({sum(len(v) for v in matrix.values())} checks) incl. separation of duties")

    # 7 masked views: support sees no model internals; analyst does; nobody sees the raw UPI ID
    sv = c.get(f"/api/staff/cases/{did}", headers=hdr(toks["support1"])).get_json()
    av = c.get(f"/api/staff/cases/{did}", headers=hdr(toks["analyst1"])).get_json()
    assert "shap_log_odds" not in sv and "p_repay" not in sv and "shap_log_odds" in av and "p_repay" in av
    assert sv["masked_vpa"] == "s***@okaxis" and VPA_A not in json.dumps([sv, av]); ok("support sees masked case; analyst sees SHAP; raw UPI ID never exposed")

    # 8 PII never stored: dump the whole DB
    dump = "\n".join(store.db.iterdump())
    assert VPA_A not in dump and VPA_B not in dump and MOB_A not in dump and MOB_B not in dump
    assert "steady.ravi" not in dump; ok("raw UPI ID and mobile number absent from database (only pseudonyms/masked)")

    # 9 audit chain + tamper detection
    assert c.get("/api/staff/audit", headers=hdr(toks["dpo1"])).get_json()["chain_intact"] is True
    denied = [r for r in store.audit_rows(500) if r["action"] == "access_denied"]; assert denied
    store.db.execute("UPDATE audit SET outcome='tampered' WHERE seq=2"); store.db.commit()
    r = c.get("/api/staff/audit", headers=hdr(toks["dpo1"])).get_json()
    assert r["chain_intact"] is False and r["first_bad_seq"] == 2; ok("audit trail: denials logged; tampering detected")
    store.db.execute("UPDATE audit SET outcome='ok' WHERE seq=2"); store.db.commit()

    # 10 contest -> staff review -> four-eyes
    assert jpost(c, f"/api/my/decisions/{did}/contest", {"reason": "my income is higher"}, ta).status_code == 200
    assert jpost(c, f"/api/staff/cases/{did}/review", {"outcome": "bogus"}, toks["support1"]).status_code == 400
    assert jpost(c, f"/api/staff/cases/{did}/review", {"outcome": "upheld", "note": "checked"}, toks["support1"]).status_code == 200
    assert jpost(c, f"/api/staff/cases/{did}/review", {"outcome": "upheld"}, toks["support1"]).status_code == 409   # same person twice
    assert jpost(c, f"/api/staff/cases/{did}/review", {"outcome": "manual_reassessment"}, toks["analyst1"]).status_code == 200
    v = c.get(f"/api/my/decisions/{did}", headers=hdr(ta)).get_json()
    assert v["status"] == "REVIEWED" and v["review"]["outcome"] == "manual_reassessment"; ok("contest + human review + four-eyes")

    # 11 grievance
    gid = jpost(c, "/api/my/grievance", {"text": "app was slow"}, ta).get_json()["id"]
    assert jpost(c, f"/api/staff/grievances/{gid}/resolve", {"resolution": "fixed"}, toks["dpo1"]).status_code == 200; ok("grievance filed and resolved")

    # 12 withdraw consent deletes derived data; export; erase
    cons = [x for x in c.get("/api/my/data", headers=hdr(tb)).get_json()["consents"]][0]
    assert jpost(c, f"/api/my/consent/{cons['id']}/revoke", {}, tb).status_code == 200
    assert c.get(f"/api/my/decisions/{rb['id']}", headers=hdr(tb)).status_code == 404
    assert jpost(c, f"/api/my/consent/{cons['id']}/revoke", {}, ta).status_code == 404            # not A's consent
    exp = c.get("/api/my/data", headers=hdr(ta)).get_json(); assert exp["decisions"]
    er = c.delete("/api/my/data", headers=hdr(ta)).get_json(); assert er["decisions_deleted"] >= 1
    assert not c.get("/api/my/data", headers=hdr(ta)).get_json()["decisions"]; ok("withdrawal, export and erasure work; cross-user revoke blocked")

    # 13 retention purge
    tc = customer(c, "9000000001"); d3 = analyse(c, tc, "mixed.x@upi").get_json()["id"]
    store.db.execute("UPDATE decisions SET expires_at='2020-01-01T00:00:00+00:00' WHERE id=?", (d3,)); store.db.commit()
    assert c.post("/api/staff/purge", headers=hdr(toks["dpo1"])).get_json()["deleted"] == 1; ok("retention: expired results purged")

    # 14 OTP lockout + logout revocation
    m = "9000000002"; jpost(c, "/api/auth/otp/request", {"mobile": m})
    codes = [jpost(c, "/api/auth/otp/verify", {"mobile": m, "otp": "000000"}).status_code for _ in range(6)]
    assert codes == [401] * 6
    t = staff(c, "support1"); assert jpost(c, "/api/logout", {}, t).status_code == 200
    assert c.get("/api/staff/cases", headers=hdr(t)).status_code == 401; ok("OTP lockout; logout revokes token")

    # 15 SHAP: exact additivity  base + sum(phi) == log-odds
    from explain import LinearShap
    ex = LinearShap(app.base.scorer); F = app.base.scorer.Fte
    for i in range(0, 60, 6):
        row = F.iloc[i]; p = app.base.scorer.predict(row)["p_repay"]
        got = ex.logit(row); p_exact = 1 / (1 + np.exp(-got))
        assert abs(p_exact - app.base.scorer.lr.predict_proba(app.base.scorer.sc.transform(F.iloc[[i]]))[0, 1]) < 1e-9
    xc = ex.crosscheck(); ok(f"SHAP additivity exact (shap-library crosscheck: {'skipped, not installed' if xc is None else f'max diff {xc:.2e}'})")

    # 16 fairness report structure
    fr = app.base and c.get("/api/staff/fairness", headers=hdr(toks["analyst1"])).get_json()
    assert set(fr["attributes"]) >= {"gender", "region", "age_band", "language", "gender_x_region"}
    assert fr["overall_verdict"].split(" (")[0] in ("PASS", "REVIEW REQUIRED", "INCOMPLETE") and fr["proxy_candidates"] is not None
    for a, v in fr["attributes"].items():
        s = v["summary"]; print(f"      {a:<16} DP-ratio={s['demographic_parity_ratio']:.2f} EO-diff={s['equalized_odds_difference']:.2f} "
                                f"AUC-gap={s['auc_gap']:.2f} -> {s['verdict']}")
    ok(f"fairness audit generated (engine: {fr['engine']}; overall: {fr['overall_verdict']})")

    # 17 prod mode: refuses default secrets; withholds unverified scripts until a human releases them
    for k in ("APP_JWT_SECRET", "APP_PEPPER", "STAFF_DEMO_PASSWORD"): os.environ.pop(k, None)
    try: create_app(env="prod", n_workers=300); raise SystemExit("prod started with default secrets!")
    except RuntimeError: pass
    os.environ.update(APP_JWT_SECRET="x" * 32, APP_PEPPER="y" * 32, STAFF_DEMO_PASSWORD=STAFF_PW)
    p = create_app(env="prod", n_workers=300); pc = p.test_client()
    assert "sandbox_otp" not in jpost(pc, "/api/auth/otp/request", {"mobile": MOB_A}).get_json()
    tok = rbac.issue_token(pseudonym(MOB_A, "mobile"), "customer")
    w = analyse(pc, tok, VPA_A); assert w.status_code == 202 and w.get_json()["status"] == "waiting_for_aa_approval"
    cid = w.get_json()["consent_id"]
    assert jpost(pc, f"/api/analyse/{cid}/continue", {}, tok).status_code == 202            # still not approved
    other = rbac.issue_token(pseudonym("9111111111", "mobile"), "customer")
    assert jpost(pc, f"/api/analyse/{cid}/continue", {}, other).status_code == 404          # someone else's pending request
    p.aa.approve(store_handle(p, cid))                                                       # customer taps approve in AA app
    r = jpost(pc, f"/api/analyse/{cid}/continue", {}, tok).get_json()
    assert r["status"] == "UNDER_REVIEW" and r["script"] is None, r
    sup = jpost(pc, "/api/staff/login", {"username": "support1", "password": STAFF_PW}).get_json()["token"]
    assert jpost(pc, f"/api/staff/cases/{r['id']}/review", {"outcome": "approve_release", "note": "facts verified"}, sup).status_code == 200
    assert pc.get(f"/api/my/decisions/{r['id']}", headers=hdr(tok)).get_json()["script"]; ok("prod: refuses default secrets, hides OTP, withholds unverified script until human release")
    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    main()
