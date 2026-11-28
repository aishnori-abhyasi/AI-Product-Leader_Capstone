"""Corrections + incident/breach workflow tests. Run: python test_incidents.py"""
import json, time
from datetime import datetime, timedelta, timezone
import i18n, incidents
from app import create_app
from test_app import customer, analyse, staff, hdr, jpost

NOTICE_EN = dict(what="Some of your saved results could be seen by an unauthorised person.", when="Between 2 and 4 November 2026.",
                 data="Your level and your list of suggested schemes. No bank password, UPI PIN or raw transactions.",
                 effect="Someone might guess your work type and income pattern.", doing="We closed the access and are checking all logs.",
                 you="Do not share OTPs with anyone. Contact us if you see unusual calls.", contact="Data Protection Officer, grievance@example.org")
NOTICE_HI = {k: "हिन्दी: " + v for k, v in NOTICE_EN.items()}

def stamp(h=0): return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat(timespec="seconds")

def main():
    ok = lambda n: print("PASS", n)
    app = create_app(env="sandbox", n_workers=400); c, store = app.test_client(), app.store
    T = {u: staff(c, u) for u in ("support1", "analyst1", "dpo1", "admin1")}
    A, B, C, D = (customer(c, f"900000002{i}") for i in range(4))

    # ============ CORRECTIONS ============
    ra = analyse(c, A, "steady.ravi@okaxis", occ="delivery rider", lang="en").get_json()
    rb = analyse(c, B, "mixed.arun@upi", occ="delivery rider", lang="hi").get_json()
    rc = analyse(c, C, "irregular.meena@ybl", occ="ride-app driver", lang="ta").get_json()
    rd = analyse(c, D, "mixed.kiran@upi", occ="home-service worker", lang="te").get_json()
    dA = ra["id"]; pA = store.get_decision(dA)["payload"]
    assert "SVANidhi" not in json.dumps(ra["next_steps"]) and pA["features"] and pA["occupation"] == "delivery rider"
    corr = lambda tok, did, value, field="occupation": jpost(c, f"/api/my/decisions/{did}/correction", {"field": field, "value": value, "reason": "I sell on the street"}, tok)
    assert corr(B, dA, "street vendor").status_code == 404                       # not B's decision
    assert corr(A, dA, "astronaut").status_code == 400 and corr(A, dA, "street vendor", field="income").status_code == 400
    assert corr(A, dA, "delivery rider").status_code == 400                       # unchanged value
    k1 = corr(A, dA, "street vendor"); assert k1.status_code == 200; k1 = k1.get_json()["id"]
    assert [x["status"] for x in c.get("/api/my/corrections", headers=hdr(A)).get_json()] == ["OPEN"]
    assert c.get("/api/my/corrections", headers=hdr(B)).get_json() == []
    rbac_get = lambda who: c.get("/api/staff/corrections", headers=hdr(T[who])).status_code
    assert (rbac_get("support1"), rbac_get("dpo1"), rbac_get("analyst1"), rbac_get("admin1")) == (200, 200, 403, 403)
    assert c.get("/api/staff/corrections", headers=hdr(A)).status_code == 403
    assert jpost(c, f"/api/staff/corrections/{k1}/resolve", {"action": "apply"}, T["dpo1"]).status_code == 403      # read-only role
    assert jpost(c, f"/api/staff/corrections/{k1}/resolve", {"action": "maybe"}, T["support1"]).status_code == 400
    assert jpost(c, f"/api/staff/corrections/{k1}/resolve", {"action": "apply", "note": "verified by phone"}, T["support1"]).status_code == 200
    p2 = store.get_decision(dA)["payload"]; rev = store.get_decision(dA)["review"]
    assert p2["occupation"] == "street vendor" and rev["corrections"][0]["to"] == "street vendor" and p2["features"] == pA["features"]
    if p2["credit_ok"]: assert "svanidhi" in [o["id"] for o in p2["offers"]]
    assert jpost(c, f"/api/staff/corrections/{k1}/resolve", {"action": "apply"}, T["support1"]).status_code == 404   # already resolved
    k2 = corr(A, dA, "ride-app driver").get_json()["id"]
    assert jpost(c, f"/api/staff/corrections/{k2}/resolve", {"action": "reject", "note": "no evidence"}, T["support1"]).status_code == 200
    assert store.get_decision(dA)["payload"]["occupation"] == "street vendor"
    assert {x["status"] for x in c.get("/api/my/corrections", headers=hdr(A)).get_json()} == {"APPLIED", "REJECTED"}
    assert len(c.get("/api/my/data", headers=hdr(A)).get_json()["corrections"]) == 2
    ok(f"corrections: own-data only, validated, applied by re-running agents on stored features (credit_ok={p2['credit_ok']}), rejected path leaves result unchanged, in data export")

    # ============ INCIDENTS ============
    good = dict(title="Unauthorised read of saved results", description="A misconfigured storage rule exposed stored results for two days.",
                kind="personal_data_breach", personal_data_involved=True, severity="HIGH", categories=["derived income features"],
                scope={"all_active": True}, cert_in_applicable=True, aware_at=stamp(2), notice={"en": NOTICE_EN, "hi": NOTICE_HI, "ta": {"what": "தமிழ்"}})
    assert jpost(c, "/api/staff/incidents", good, A).status_code == 403                                           # customers cannot declare
    assert jpost(c, "/api/staff/incidents", {**good, "severity": "EPIC"}, T["support1"]).status_code == 400
    assert jpost(c, "/api/staff/incidents", {**good, "aware_at": stamp(-5)}, T["support1"]).status_code == 400    # future
    assert jpost(c, "/api/staff/incidents", {**good, "notice": {"en": {"what": "x"}}}, T["support1"]).status_code == 400
    r = jpost(c, "/api/staff/incidents", good, T["support1"]); assert r.status_code == 201; iid = r.get_json()["id"]
    assert jpost(c, "/api/staff/incidents", {**good, "title": "second"}, T["admin1"]).status_code == 201          # admin may also declare
    for who, code in (("support1", 403), ("analyst1", 403), ("admin1", 403), ("dpo1", 200)):
        assert c.get("/api/staff/incidents", headers=hdr(T[who])).status_code == code, who
    g = c.get(f"/api/staff/incidents/{iid}", headers=hdr(T["dpo1"])).get_json(); cl = g["clocks"]
    get = lambda pre: next(v for k, v in cl.items() if k.startswith(pre))
    aware = datetime.fromisoformat(g["aware_at"])
    assert datetime.fromisoformat(get("board_report_72h")["due"]) == aware + timedelta(hours=72)
    assert datetime.fromisoformat(get("cert_in_6h")["due"]) == aware + timedelta(hours=6) and get("cert_in_6h")["status"] == "PENDING"
    assert 3.5 < get("cert_in_6h")["hours_left"] < 4.5 and cl["principals"]["status"] == "PENDING"
    late = jpost(c, "/api/staff/incidents", {**good, "title": "old", "aware_at": stamp(8)}, T["support1"]).get_json()["id"]
    assert [v for k, v in c.get(f"/api/staff/incidents/{late}", headers=hdr(T["dpo1"])).get_json()["clocks"].items() if k.startswith("cert_in")][0]["status"] == "OVERDUE"
    ok("incident declaration: roles, validation, statutory clocks (72h Board report, 6h CERT-In) and overdue detection")

    # notices
    assert jpost(c, f"/api/staff/incidents/{iid}/notify-principals", {}, T["support1"]).status_code == 403
    n = jpost(c, f"/api/staff/incidents/{iid}/notify-principals", {}, T["dpo1"]).get_json()
    assert n["created"] == 4 and n["english_fallback"] == 2, n            # A(en) ok, B(hi) ok, C(ta partial)+D(te none) fall back
    assert jpost(c, f"/api/staff/incidents/{iid}/notify-principals", {}, T["dpo1"]).get_json() == {"created": 0, "english_fallback": 0, "already_notified": 4}
    nb = c.get("/api/my/notifications", headers=hdr(B)).get_json()[0]; nc = c.get("/api/my/notifications", headers=hdr(C)).get_json()[0]
    assert nb["lang"] == "hi" and nb["labels"]["what"] == i18n.UI["hi"]["b_what"] and nb["fields"]["what"].startswith("हिन्दी")
    assert nc["lang"] == "en" and nc["labels"]["what"] == "What happened" and nc["fields"]["what"] == NOTICE_EN["what"]
    assert set(nb["fields"]) == set(incidents.NOTICE_FIELDS) and not nb["read"]
    assert jpost(c, f"/api/my/notifications/{nb['id']}/read", {}, A).status_code == 404 and jpost(c, f"/api/my/notifications/{nb['id']}/read", {}, B).status_code == 200
    assert c.get("/api/my/notifications", headers=hdr(B)).get_json()[0]["read"] is True; ok("Rule-7 style notices: in each person's language, English fallback flagged, read-tracking, idempotent, own-data only")

    # erased people are not notified; non-personal incidents have no principal notices
    E = customer(c, "9000000029"); analyse(c, E, "steady.erased@okaxis"); c.delete("/api/my/data", headers=hdr(E))
    i2 = jpost(c, "/api/staff/incidents", {**good, "title": "second wave"}, T["support1"]).get_json()["id"]
    assert jpost(c, f"/api/staff/incidents/{i2}/notify-principals", {}, T["dpo1"]).get_json()["created"] == 4
    assert c.get("/api/my/notifications", headers=hdr(E)).get_json() == []
    np_ = jpost(c, "/api/staff/incidents", {**good, "title": "outage", "kind": "availability_incident", "personal_data_involved": False, "cert_in_applicable": False, "notice": None}, T["support1"]).get_json()["id"]
    assert jpost(c, f"/api/staff/incidents/{np_}/notify-principals", {}, T["dpo1"]).status_code == 400; ok("erased users excluded; no notices for incidents without personal data")

    # board report + closing gates
    br = c.get(f"/api/staff/incidents/{iid}/board-report", headers=hdr(T["dpo1"])).get_json()
    assert br["status"].startswith("DRAFT") and br["detailed_report_72h"]["intimations_to_data_principals"]["delivered_in_app"] == 4 and br["detailed_report_72h"]["intimations_to_data_principals"]["read"] == 1
    assert c.get(f"/api/staff/incidents/{iid}/board-report", headers=hdr(T["support1"])).status_code == 403
    close = lambda pm="reviewed": jpost(c, f"/api/staff/incidents/{iid}/status", {"status": "CLOSED", "post_mortem": pm}, T["dpo1"]).status_code
    assert close() == 409                                                                  # nothing filed yet
    for ev in ("board_initial", "board_report"):
        assert jpost(c, f"/api/staff/incidents/{iid}/record", {"event": ev, "reference": "ref-1"}, T["dpo1"]).status_code == 200
    assert jpost(c, f"/api/staff/incidents/{iid}/record", {"event": "board_report"}, T["dpo1"]).status_code == 400   # once only
    assert close() == 409                                                                  # CERT-In still missing
    jpost(c, f"/api/staff/incidents/{iid}/record", {"event": "cert_in", "reference": "cert-ref"}, T["dpo1"])
    assert close("") == 409                                                                # needs a post-incident review
    assert close("Root cause: storage rule. Fix: policy-as-code check.") == 200
    done = c.get(f"/api/staff/incidents/{iid}", headers=hdr(T["dpo1"])).get_json()
    assert done["status"] == "CLOSED" and all(v["status"] == "DONE" for k, v in done["clocks"].items() if k.startswith(("board", "cert"))); ok("Board-report draft counts deliveries; incident cannot close without Board/CERT-In/principal steps and a post-incident review")

    # kill switch
    F_ = customer(c, "9000000030")
    assert jpost(c, "/api/staff/system/pause", {"paused": True}, T["support1"]).status_code == 403 and jpost(c, "/api/staff/system/pause", {"paused": True}, T["analyst1"]).status_code == 403
    assert jpost(c, "/api/staff/system/pause", {"paused": True, "reason": "incident"}, T["dpo1"]).status_code == 200
    assert analyse(c, F_, "steady.paused@okaxis").status_code == 503
    assert jpost(c, "/api/my/decisions/x/contest", {}, F_).status_code == 404                  # other customer features stay available
    assert jpost(c, "/api/staff/system/pause", {"paused": False}, T["admin1"]).status_code == 200
    assert analyse(c, F_, "steady.paused@okaxis").status_code == 200; ok("kill switch: DPO/admin only; blocks new analyses with 503; resumable")

    # revoke every session
    assert jpost(c, "/api/staff/system/revoke-sessions", {}, T["support1"]).status_code == 403
    assert jpost(c, "/api/staff/system/revoke-sessions", {}, T["admin1"]).status_code == 200
    assert c.get("/api/my/decisions", headers=hdr(A)).status_code == 401 and c.get("/api/staff/incidents", headers=hdr(T["dpo1"])).status_code == 401
    time.sleep(0.02); A2 = customer(c, "9000000020"); dpo2 = staff(c, "dpo1")
    assert c.get("/api/my/decisions", headers=hdr(A2)).status_code == 200 and c.get("/api/staff/incidents", headers=hdr(dpo2)).status_code == 200
    ok("revoke-all-sessions: old customer and staff tokens die, fresh logins work")

    # audit + PII
    assert store.verify_audit()[0]
    acts = {r["action"] for r in store.audit_rows(2000)}
    need = {"incident_declared", "principals_notified", "incident_event", "incident_status", "processing_paused", "processing_resumed", "all_sessions_revoked", "correction_requested", "correction_applied", "correction_rejected", "notification_read", "board_report_drafted"}
    assert need <= acts, need - acts
    dump = "\n".join(store.db.iterdump())
    for raw in ("steady.ravi", "mixed.arun", "irregular.meena", "9000000020", "9000000021"): assert raw not in dump
    ok("every step audited, chain intact, no raw UPI IDs or mobile numbers anywhere in the database")
    print("\nALL INCIDENT/CORRECTION TESTS PASSED")

if __name__ == "__main__": main()
