"""
UtiliSave customer app + staff API.

    pip install flask pyjwt scikit-learn pandas numpy langchain-core langgraph langchain-anthropic shap fairlearn
    python app.py                       # http://127.0.0.1:5000  (sandbox, synthetic data, demo OTP shown on screen)

Customer flow:  mobile OTP -> enter UPI ID + occupation -> read notice, tick consent
                -> (sandbox) Account Aggregator approval -> 4 agents run in the background
                -> plain-English suggestion + simple reasons + "how to improve" + contest / delete buttons.
ENV=prod refuses to start with default secrets and WITHHOLDS any script that is not safe_to_ship
(unverified scheme facts, readability or grounding failure) until a staff member releases it.
"""
import json, os, secrets, time

import pandas as pd
from flask import Flask, g, jsonify, request

import i18n
import incidents
import rbac
from aa_gateway import SandboxAA
from compliance import (DATA_LIFE_DAYS, FI_DAYS, NOTICE_VERSION, PURPOSE, Store, mask_vpa, notice, pseudonym,
                        valid_mobile, valid_vpa)
from explain import LinearShap
from fairness import MODEL_VERSION, model_card, run_audit
from matching_engine import FEATURES
from upi_agents_mvp import Context, build_context, build_pipeline, engineer_features, make_llm

OCCUPATIONS = ["delivery rider", "ride-app driver", "street vendor", "home-service worker", "other"]

from compliance import iso, now

def iso_now(): return iso(now())

def create_app(env=None, db=None, n_workers=1500, aa=None):
    env = env or os.getenv("ENV", "sandbox")
    db = db or os.getenv("APP_DB", ":memory:")           # set APP_DB=/data/utilisave.db to persist (Docker volume)
    if env == "prod":
        missing = [k for k in ("APP_JWT_SECRET", "APP_PEPPER", "STAFF_DEMO_PASSWORD") if not os.getenv(k)]
        if missing: raise RuntimeError(f"refusing to start in prod without: {missing}")
    app = Flask(__name__)
    store, limiter, aa, llm = Store(db), rbac.Limiter(), aa or SandboxAA(), make_llm()
    aa_mode = "sandbox" if isinstance(aa, SandboxAA) else "live"
    rbac.init(store)
    base = build_context(n_workers)
    explainer = LinearShap(base.scorer)
    fairness = run_audit(base)
    card = model_card(base, fairness, explainer.global_importance())
    otps, pending = {}, {}
    pw = os.getenv("STAFF_DEMO_PASSWORD", "Demo-Staff-Pass-1")
    staff = {u: {"role": r, "pw": rbac.hash_password(pw)} for u, r in
             {"support1": "support", "analyst1": "risk_analyst", "dpo1": "compliance_officer", "admin1": "admin"}.items()}

    # ---------------------------------------------------------------- helpers
    def actor(): return g.claims["sub"]
    def role(): return g.claims["role"]

    def customer_view(d):
        p, rev = d["payload"], d["review"]; lang = p.get("lang", "en")
        out = {"id": d["id"], "status": d["status"], "lang": lang, "level": i18n.LEVELS[lang][p["band"]], "expires_at": d["expires_at"],
               "disclosure": i18n.DISCLOSURE[lang], "review": ({"outcome": rev.get("outcome"), "note": rev.get("note")} if rev and "outcome" in rev else None)}
        if p["release"]:
            out.update({"headline": i18n.HEADLINE[lang]["credit" if p["credit_ok"] else "no_credit"], "script": p["script"],
                        "next_steps": [{"name": i18n.offer_name(lang, o), "action": i18n.KIND_VERB[lang].get(o["kind"], "")} for o in p["offers"]],
                        "why": [{"text": r["text"] if lang == "en" else i18n.REASON[lang][r["feature"]][0 if r["effect"] == "helps" else 1],
                                 "effect": i18n.EFFECT[lang]["helps" if r["effect"] == "helps" else "holds_back"]} for r in p["reasons_plain"]],
                        "how_to_improve": [x["text"] if lang == "en" else i18n.RECOURSE[lang][(x["feature"], x["enough"])] for x in p["recourse"]]})
            if p.get("imputed_features"): out["data_note"] = i18n.DATA_NOTE[lang]
            if env != "prod" and not p["safe_to_ship"]:
                out["demo_notice"] = "DEMO: scheme facts" + ("" if lang == "en" or i18n.REVIEW.get(lang) else " and this translation") + " are not yet verified. Do not rely on them."
        else:
            out.update({"headline": "Your suggestion is being checked by our team.", "script": None, "next_steps": [], "why": [], "how_to_improve": []})
        return out

    def staff_view(d, with_explain):
        p = d["payload"]
        v = {"id": d["id"], "status": d["status"], "created_at": d["created_at"], "masked_vpa": d["masked_vpa"],
             "level": i18n.LEVELS["en"][p["band"]], "credit_ok": p["credit_ok"], "offers": [o["id"] for o in p["offers"]],
             "released": p["release"], "review": d["review"]}
        if with_explain:
            v.update({"segment": p["segment"], "p_repay": p["p_repay"], "shap_log_odds": p["shap"],
                      "agent_trail": p["agent_trail"], "safe_to_ship": p["safe_to_ship"], "model_version": d["model_version"]})
        return v

    def own(did):
        d = store.get_decision(did)
        return d if d and d["owner"] == actor() else None        # object-level authorisation (no IDOR)

    # ---------------------------------------------------------------- web page + notice
    @app.get("/")
    def index():
        nonce = secrets.token_urlsafe(16)
        g.nonce = nonce
        import html as _h, re as _re
        page = _re.sub(r'(data-i="(\w+)"[^>]*>)(</)', lambda m: m.group(1) + _h.escape(i18n.UI["en"][m.group(2)]) + m.group(3), PAGE)   # readable even if JS is blocked
        page = page.replace('<select id="occ"></select>', '<select id="occ">' + "".join(f'<option value="{k}">{v}</option>' for k, v in i18n.OCC["en"].items()) + '</select>')
        return page.replace("__NONCE__", nonce)

    @app.get("/api/notice")
    def get_notice(): return jsonify(i18n.notice_for(i18n.norm(request.args.get("lang")), notice()))

    @app.get("/api/i18n/<lang>")
    def i18n_get(lang):
        if lang not in i18n.LANGS: return jsonify(error="unsupported language"), 404
        return jsonify(ui=i18n.UI[lang], occupations=i18n.OCC[lang], languages=i18n.LANGS)

    @app.get("/healthz")
    def health(): return jsonify(status="ok", env=env, agents_use_llm=llm is not None, aa_mode=aa_mode, needs_aa_id=aa_mode != "sandbox")

    @app.after_request
    def headers(r):
        n = getattr(g, "nonce", None)
        r.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; " +
                                                (f"script-src 'nonce-{n}'" if n else "script-src 'none'"))
        r.headers["X-Content-Type-Options"] = "nosniff"; r.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api"): r.headers["Cache-Control"] = "no-store"
        return r

    # ---------------------------------------------------------------- customer auth (mobile OTP)
    @app.post("/api/auth/otp/request")
    def otp_request():
        mobile = (request.get_json(silent=True) or {}).get("mobile", "")
        if not valid_mobile(mobile): return jsonify(error="enter a valid 10-digit mobile number"), 400
        owner = pseudonym(mobile, "mobile")
        if limiter.blocked("otp:" + owner, 3, 600) or limiter.blocked("ip:" + str(request.remote_addr), 20, 600):
            return jsonify(error="too many requests, try later"), 429
        otp = f"{secrets.randbelow(10 ** 6):06d}"
        otps[owner] = {"h": pseudonym(otp, owner), "exp": time.time() + 300, "tries": 0}
        store.audit(owner, "customer", "otp_requested")
        return jsonify(sent=True, **({"sandbox_otp": otp} if env != "prod" else {}))      # real SMS in prod

    @app.post("/api/auth/otp/verify")
    def otp_verify():
        b = request.get_json(silent=True) or {}
        owner = pseudonym(b.get("mobile", ""), "mobile")
        rec = otps.get(owner)
        if not rec or rec["exp"] < time.time() or rec["tries"] >= 5:
            return jsonify(error="code expired or locked, request a new one"), 401
        rec["tries"] += 1
        if not secrets.compare_digest(rec["h"], pseudonym(str(b.get("otp", "")), owner)):
            store.audit(owner, "customer", "otp_failed", outcome="denied"); return jsonify(error="wrong code"), 401
        otps.pop(owner, None)
        store.audit(owner, "customer", "login")
        return jsonify(token=rbac.issue_token(owner, "customer"))

    # ---------------------------------------------------------------- customer: analyse
    @app.post("/api/analyse")
    @rbac.require("self:analyse")
    def analyse():
        b = request.get_json(silent=True) or {}
        vpa, occ = (b.get("vpa") or "").strip(), b.get("occupation", "other")
        if not valid_vpa(vpa) or occ not in OCCUPATIONS: return jsonify(error="check your UPI ID and occupation"), 400
        if paused(): return jsonify(error="The service is paused for maintenance. Please try again later."), 503
        if b.get("adult") is not True: return jsonify(error="this service is for people aged 18 or over"), 400
        if b.get("consent") is not True or b.get("notice_version") != NOTICE_VERSION:
            return jsonify(error="please read the notice and give consent to continue"), 400
        if limiter.blocked("an:" + actor(), 10, 3600): return jsonify(error="too many checks, try later"), 429
        owner, subject, lang = actor(), pseudonym(vpa, "vpa"), i18n.norm(b.get("lang"))
        try:
            handle = aa.request_consent(subject, PURPOSE, FI_DAYS, DATA_LIFE_DAYS, hint=vpa.split("@")[0], aa_id=(b.get("aa_id") or "").strip() or None)
        except ValueError:
            return jsonify(error="enter your Account Aggregator ID like 9876543210@provider"), 400
        except Exception:
            app.logger.exception("AA consent request failed")
            return jsonify(error="the Account Aggregator is not reachable right now, please try again"), 502
        cid = store.grant_consent(owner, subject, handle, language=lang)
        store.audit(owner, "customer", "consent_granted", cid, notice=NOTICE_VERSION, purpose=PURPOSE, language=lang)
        pending[cid] = {"owner": owner, "subject": subject, "handle": handle, "masked": mask_vpa(vpa), "occ": occ, "lang": lang}   # no raw UPI ID kept
        if hasattr(aa, "approve") and env != "prod": aa.approve(handle)      # sandbox only; live: customer approves in their AA app
        return run_pending(cid)

    @app.post("/api/analyse/<cid>/continue")
    @rbac.require("self:analyse")
    def analyse_continue(cid):
        """Called after the customer approves in their AA app (or from the AA consent-notification webhook)."""
        if cid not in pending or pending[cid]["owner"] != actor(): return jsonify(error="not found"), 404
        return run_pending(cid)

    def build_payload(subject, F, occ, lang, imputed, days_of_data=FI_DAYS):
        """Run the 4 agents for one person from a one-row feature frame. Used for the first run AND for re-runs after a correction."""
        meta = pd.DataFrame({"archetype": ["n/a"], "occupation": [occ], "has_bank_account": [True], "consent": [True],
                             "days_of_data": [days_of_data]}, index=[subject])
        state = build_pipeline(Context(F, meta, base.seg, base.scorer, base.matcher, base.retriever, lang=lang), llm)[0] \
            .invoke({"worker_id": subject, "audit": []})
        if state.get("blocked"): return None, state["blocked"]
        row, m, s, c = F.iloc[0], state["matching"], state["scoring"], state["communication"]
        release = env != "prod" or c["safe_to_ship"]
        return {"segment": m["segment"], "band": m["band"], "p_repay": s["p_repay"], "credit_ok": m["credit_ok"],
                "offers": m["offers"], "script": c["script"], "grade": c["grade"], "safe_to_ship": c["safe_to_ship"],
                "shap": {k: round(v, 4) for k, v in explainer.values(row).items()},
                "reasons_plain": explainer.plain_reasons(row), "imputed_features": imputed,
                "recourse": explainer.recourse(row, s["p_repay"], base.scorer.BANDS),
                "lang": lang, "readability": c.get("readability"), "translation_reviewed": c.get("translation_reviewed"),
                "occupation": occ, "days_of_data": days_of_data, "features": {k: float(row[k]) for k in FEATURES},
                "agent_trail": [{"agent": e["agent"], "tool": e["tool"], "explanation": e["explanation"], "llm_used": e["llm_used"]}
                                for e in state["audit"]], "release": release}, None

    def paused(): return bool(store.get_flag("processing_paused", False))

    def run_pending(cid):
        if paused(): return jsonify(error="The service is paused for maintenance. Please try again later."), 503
        pd_ = pending[cid]
        if not store.consent_is_active(cid, pd_["owner"]): pending.pop(cid, None); return jsonify(error="consent is no longer active"), 403
        if aa.status(pd_["handle"]) != "APPROVED":
            return jsonify(status="waiting_for_aa_approval", consent_id=cid, message="Approve the request in your Account Aggregator app, then tap Continue."), 202
        pending.pop(cid)
        owner, subject, occ, lang = pd_["owner"], pd_["subject"], pd_["occ"], pd_["lang"]
        tel = aa.fetch(pd_["handle"])      # in memory only
        F = engineer_features(tel)
        for k, v in tel.attrs.get("feature_overrides", {}).items(): F[k] = v
        imputed = []
        if tel.attrs.get("no_bill_signal"):             # no visible bills is missing data, not bad behaviour -> neutral value
            F["bill_on_time_ratio"] = float(base.scorer.sc.mean_[FEATURES.index("bill_on_time_ratio")]); imputed.append("bill_on_time_ratio")
        days = int(tel["day"].nunique()); del tel
        payload, blocked = build_payload(subject, F, occ, lang, imputed, days)
        if blocked:
            store.audit(owner, "customer", "analysis_blocked", cid, reason=blocked)
            return jsonify(error="We could not check your options: " + blocked), 422
        did = store.save_decision(owner, subject, cid, pd_["masked"], MODEL_VERSION, "OK" if payload["release"] else "UNDER_REVIEW", payload)
        store.audit(owner, "customer", "decision_created", did, band=payload["band"], credit_ok=payload["credit_ok"], released=payload["release"])
        return jsonify(customer_view(store.get_decision(did)))

    @app.post("/api/aa/notify/consent")
    def aa_notify():
        """Account Aggregator -> us. Signature is verified before anything else; unsigned calls change nothing."""
        raw = request.get_data()
        if not hasattr(aa, "verify_notification") or not aa.verify_notification(raw, request.headers.get("x-jws-signature")):
            store.audit("aa", "external", "aa_notification_rejected", outcome="denied"); return jsonify(error="invalid signature"), 401
        try: n = json.loads(raw).get("ConsentStatusNotification", {})
        except ValueError: return jsonify(error="bad body"), 400
        aa.handle_notification(json.loads(raw))
        store.audit("aa", "external", "aa_consent_notification", n.get("consentHandle", ""), status=n.get("consentStatus"))
        if n.get("consentStatus") in ("REVOKED", "REJECTED", "PAUSED", "EXPIRED"):   # withdrawal at the AA = withdrawal here
            c = store.consent_by_handle(n.get("consentHandle"))
            if c and c["status"] == "ACTIVE":
                res = store.revoke_consent(c["id"], c["owner"]); pending.pop(c["id"], None)
                store.audit(c["owner"], "customer", "consent_withdrawn_at_aa", c["id"], decisions_deleted=res["decisions_deleted"])
        return jsonify(ok=True)

    # ---------------------------------------------------------------- customer: own data & rights
    @app.get("/api/my/decisions")
    @rbac.require("self:read")
    def my_decisions(): return jsonify([customer_view(d) for d in store.list_decisions(owner=actor())])

    @app.get("/api/my/decisions/<did>")
    @rbac.require("self:read")
    def my_decision(did):
        d = own(did)
        if not d: return jsonify(error="not found"), 404
        return jsonify(customer_view(d))

    @app.post("/api/my/decisions/<did>/contest")
    @rbac.require("self:contest")
    def contest(did):
        d = own(did)
        if not d: return jsonify(error="not found"), 404
        reason = ((request.get_json(silent=True) or {}).get("reason") or "")[:1000]
        store.update_decision(did, status="CONTESTED", review={"contest_reason": reason})
        store.audit(actor(), "customer", "decision_contested", did)
        return jsonify(status="CONTESTED", message="A person will review this. We will update this page.")

    @app.post("/api/my/consent/<cid>/revoke")
    @rbac.require("self:consent")
    def revoke(cid):
        res = store.revoke_consent(cid, actor())
        if not res: return jsonify(error="not found"), 404
        aa.revoke(res["aa_handle"])
        store.audit(actor(), "customer", "consent_withdrawn", cid, decisions_deleted=res["decisions_deleted"])
        return jsonify(status="WITHDRAWN", deleted=res["decisions_deleted"])

    @app.get("/api/my/data")
    @rbac.require("self:export")
    def export():
        store.audit(actor(), "customer", "data_exported")
        return jsonify(store.export_owner(actor()))

    @app.delete("/api/my/data")
    @rbac.require("self:erase")
    def erase():
        res = store.erase_owner(actor())
        store.audit(actor(), "customer", "data_erased", **res)
        return jsonify(res)

    @app.post("/api/my/grievance")
    @rbac.require("self:grievance")
    def grievance():
        text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
        if not text: return jsonify(error="please describe the problem"), 400
        gid = store.add_grievance(actor(), text)
        store.audit(actor(), "customer", "grievance_filed", gid)
        return jsonify(id=gid)

    # ---------------------------------------------------------------- staff
    @app.post("/api/staff/login")
    def staff_login():
        b = request.get_json(silent=True) or {}
        u = staff.get(b.get("username", ""))
        if limiter.blocked("st:" + str(b.get("username")), 5, 300): return jsonify(error="too many attempts"), 429
        if not u or not rbac.check_password(b.get("password", ""), u["pw"]):
            store.audit("staff:" + str(b.get("username"))[:30], "unknown", "staff_login_failed", outcome="denied")
            return jsonify(error="invalid credentials"), 401
        store.audit("staff:" + b["username"], u["role"], "staff_login")
        return jsonify(token=rbac.issue_token("staff:" + b["username"], u["role"]), role=u["role"])

    @app.post("/api/logout")
    @rbac.require()
    def logout(): rbac.revoke_current(); return jsonify(ok=True)

    @app.get("/api/staff/cases")
    @rbac.require("case:read_masked")
    def cases():
        with_x = "explain:read_case" in rbac.PERMISSIONS[role()]
        return jsonify([staff_view(d, with_x) for d in store.list_decisions(status=request.args.get("status"))])

    @app.get("/api/staff/cases/<did>")
    @rbac.require("case:read_masked")
    def case(did):
        d = store.get_decision(did)
        if not d: return jsonify(error="not found"), 404
        store.audit(actor(), role(), "case_viewed", did)
        return jsonify(staff_view(d, "explain:read_case" in rbac.PERMISSIONS[role()]))

    @app.post("/api/staff/cases/<did>/review")
    @rbac.require("case:review")
    def review(did):
        d = store.get_decision(did)
        if not d: return jsonify(error="not found"), 404
        b = request.get_json(silent=True) or {}
        outcome = b.get("outcome")
        if outcome not in ("approve_release", "upheld", "manual_reassessment"): return jsonify(error="invalid outcome"), 400
        if d["review"] and d["review"].get("reviewer") == actor(): return jsonify(error="four-eyes: a different reviewer is required"), 409
        rev = {**(d["review"] or {}), "outcome": outcome, "note": (b.get("note") or "")[:1000], "reviewer": actor()}
        if outcome == "approve_release":
            d["payload"]["release"] = True; store.update_decision(did, status="OK", review=rev, payload=d["payload"])
        else:
            store.update_decision(did, status="REVIEWED", review=rev)
        store.audit(actor(), role(), "case_reviewed", did, outcome=outcome)
        return jsonify(ok=True)

    @app.get("/api/staff/model-report")
    @rbac.require("model:read_report")
    def model_report(): return jsonify({"model_card": card, "shap_crosscheck_vs_shap_lib": explainer.crosscheck()})

    @app.get("/api/staff/fairness")
    @rbac.require("fairness:read")
    def fairness_report(): return jsonify(fairness)


    # ---------------------------------------------------------------- corrections (customer)
    @app.post("/api/my/decisions/<did>/correction")
    @rbac.require("self:correct")
    def correction_create(did):
        d = own(did)
        if not d: return jsonify(error="not found"), 404
        b = request.get_json(silent=True) or {}
        if b.get("field") != "occupation" or b.get("value") not in OCCUPATIONS:
            return jsonify(error="you can correct your occupation here; for other problems use 'ask a person to review' or file a complaint"), 400
        if b["value"] == d["payload"].get("occupation"): return jsonify(error="that is already your occupation"), 400
        if len([k for k in store.list_corrections(owner=actor(), status="OPEN") if k["decision_id"] == did]) >= 3:
            return jsonify(error="you already have corrections waiting for this result"), 429
        cid = store.add_correction(actor(), did, "occupation", d["payload"].get("occupation", ""), b["value"], b.get("reason") or "")
        store.audit(actor(), "customer", "correction_requested", cid, decision=did, field="occupation")
        return jsonify(id=cid, status="OPEN")

    @app.get("/api/my/corrections")
    @rbac.require("self:correct")
    def my_corrections():
        return jsonify([{k: x[k] for k in ("id", "decision_id", "field", "new_value", "status", "created_at", "resolved_at", "note")}
                        for x in store.list_corrections(owner=actor())])

    # ---------------------------------------------------------------- notifications (customer)
    @app.get("/api/my/notifications")
    @rbac.require("self:notifications")
    def my_notifications():
        out = []
        for n in store.list_notifications(actor()):
            b = n["body"]; lang = b["lang"]
            out.append({"id": n["id"], "created_at": n["created_at"], "read": n["read_at"] is not None, "severity": b["severity"], "lang": lang,
                        "labels": {k: i18n.UI[lang][("b_" + k)] for k in incidents.NOTICE_FIELDS}, "fields": b["fields"],
                        "title": i18n.UI[lang]["inbox"]})
        return jsonify(out)

    @app.post("/api/my/notifications/<nid>/read")
    @rbac.require("self:notifications")
    def notification_read(nid):
        if not store.mark_notification_read(nid, actor()): return jsonify(error="not found"), 404
        store.audit(actor(), "customer", "notification_read", nid)
        return jsonify(ok=True)

    # ---------------------------------------------------------------- corrections (staff)
    @app.get("/api/staff/corrections")
    @rbac.require("correction:read")
    def staff_corrections():
        return jsonify([{k: x[k] for k in ("id", "decision_id", "field", "old_value", "new_value", "reason", "status", "created_at", "resolved_at", "resolver", "note")}
                        for x in store.list_corrections(status=request.args.get("status"))])

    @app.post("/api/staff/corrections/<kid>/resolve")
    @rbac.require("correction:process")
    def resolve_correction(kid):
        k = store.get_correction(kid)
        if not k or k["status"] != "OPEN": return jsonify(error="not found or already resolved"), 404
        b = request.get_json(silent=True) or {}
        if b.get("action") not in ("apply", "reject"): return jsonify(error="action must be apply or reject"), 400
        note = (b.get("note") or "")[:500]
        if b["action"] == "reject":
            store.resolve_correction(kid, "REJECTED", actor(), note); store.audit(actor(), role(), "correction_rejected", kid)
            return jsonify(ok=True, status="REJECTED")
        d = store.get_decision(k["decision_id"])
        if not d or "features" not in d["payload"]: return jsonify(error="the result is no longer available (deleted or expired)"), 409
        p = d["payload"]
        F = pd.DataFrame([p["features"]], index=[d["subject"]])
        payload, blocked = build_payload(d["subject"], F, k["new_value"], p["lang"], p.get("imputed_features", []), p.get("days_of_data", FI_DAYS))
        if blocked: return jsonify(error=blocked), 422
        rev = {**(d["review"] or {}), "corrections": (d["review"] or {}).get("corrections", []) + [
            {"id": kid, "field": "occupation", "from": k["old_value"], "to": k["new_value"], "applied_by": actor(), "at": iso_now()}]}
        store.update_decision(d["id"], status="OK" if payload["release"] else "UNDER_REVIEW", review=rev, payload=payload)
        store.resolve_correction(kid, "APPLIED", actor(), note)
        store.audit(actor(), role(), "correction_applied", kid, decision=d["id"], field="occupation")
        return jsonify(ok=True, status="APPLIED")

    # ---------------------------------------------------------------- incidents + kill switch (staff)
    def _inc(iid):
        inc = store.get_incident(iid)
        return inc

    @app.post("/api/staff/incidents")
    @rbac.require("incident:declare")
    def incident_declare():
        b = request.get_json(silent=True) or {}
        try:
            inc = incidents.declare(store, actor(), role(), title=b.get("title", ""), description=b.get("description", ""), kind=b.get("kind", ""),
                                    personal_data_involved=b.get("personal_data_involved", False), severity=b.get("severity", ""),
                                    categories=b.get("categories"), scope=b.get("scope"), cert_in_applicable=b.get("cert_in_applicable", False),
                                    aware_at=b.get("aware_at"), notice=b.get("notice"))
        except ValueError as ex:
            return jsonify(error=str(ex)), 400
        return jsonify(incidents.public_view(inc)), 201

    @app.get("/api/staff/incidents")
    @rbac.require("incident:read")
    def incident_list(): return jsonify([incidents.public_view(i) for i in store.list_incidents()])

    @app.get("/api/staff/incidents/<iid>")
    @rbac.require("incident:read")
    def incident_get(iid):
        inc = _inc(iid)
        if not inc: return jsonify(error="not found"), 404
        return jsonify({**incidents.public_view(inc), "notifications": store.notification_stats(iid),
                        "affected_people_in_scope": len(store.owners_in_scope(inc["scope"]))})

    @app.post("/api/staff/incidents/<iid>/notify-principals")
    @rbac.require("incident:manage")
    def incident_notify(iid):
        inc = _inc(iid)
        if not inc: return jsonify(error="not found"), 404
        try: return jsonify(incidents.notify_principals(store, inc, actor(), role()))
        except ValueError as ex: return jsonify(error=str(ex)), 400

    @app.post("/api/staff/incidents/<iid>/record")
    @rbac.require("incident:manage")
    def incident_record(iid):
        inc = _inc(iid)
        if not inc: return jsonify(error="not found"), 404
        b = request.get_json(silent=True) or {}
        try: incidents.record_event(store, inc, b.get("event", ""), actor(), role(), b.get("reference", ""), b.get("note", ""))
        except ValueError as ex: return jsonify(error=str(ex)), 400
        return jsonify(incidents.public_view(inc))

    @app.post("/api/staff/incidents/<iid>/status")
    @rbac.require("incident:manage")
    def incident_status(iid):
        inc = _inc(iid)
        if not inc: return jsonify(error="not found"), 404
        b = request.get_json(silent=True) or {}
        try: incidents.set_status(store, inc, b.get("status", ""), actor(), role(), b.get("post_mortem"))
        except ValueError as ex: return jsonify(error=str(ex)), 400
        except PermissionError as ex: return jsonify(error=str(ex)), 409
        return jsonify(incidents.public_view(inc))

    @app.get("/api/staff/incidents/<iid>/board-report")
    @rbac.require("incident:manage")
    def incident_board_report(iid):
        inc = _inc(iid)
        if not inc: return jsonify(error="not found"), 404
        store.audit(actor(), role(), "board_report_drafted", iid)
        return jsonify(incidents.board_report(store, inc))

    @app.post("/api/staff/system/pause")
    @rbac.require("system:pause")
    def system_pause():
        b = request.get_json(silent=True) or {}
        store.set_flag("processing_paused", bool(b.get("paused")))
        store.audit(actor(), role(), "processing_paused" if b.get("paused") else "processing_resumed", reason=str(b.get("reason", ""))[:200])
        return jsonify(paused=bool(b.get("paused")))

    @app.post("/api/staff/system/revoke-sessions")
    @rbac.require("system:revoke_sessions")
    def system_revoke():
        rbac.revoke_all_sessions(); store.audit(actor(), role(), "all_sessions_revoked")
        return jsonify(ok=True, message="every token issued so far is invalid; everyone must log in again")

    @app.get("/api/staff/audit")
    @rbac.require("audit:read")
    def audit_log():
        ok, bad = store.verify_audit()
        return jsonify({"chain_intact": ok, "first_bad_seq": bad, "entries": store.audit_rows(int(request.args.get("limit", 100)))})

    @app.get("/api/staff/consents")
    @rbac.require("consent:read")
    def consents(): return jsonify(store.list_consents())

    @app.get("/api/staff/grievances")
    @rbac.require("grievance:read")
    def grievances(): return jsonify(store.list_grievances())

    @app.post("/api/staff/grievances/<gid>/resolve")
    @rbac.require("grievance:resolve")
    def resolve(gid):
        ok = store.resolve_grievance(gid, (request.get_json(silent=True) or {}).get("resolution", ""))
        store.audit(actor(), role(), "grievance_resolved", gid)
        return (jsonify(ok=True) if ok else (jsonify(error="not found"), 404))

    @app.post("/api/staff/purge")
    @rbac.require("erasure:process")
    def purge():
        n = store.purge_expired(); store.audit(actor(), role(), "retention_purge", deleted=n); return jsonify(deleted=n)

    @app.get("/api/staff/users")
    @rbac.require("user:manage")
    def users(): return jsonify([{"username": u, "role": v["role"]} for u, v in staff.items()])

    app.store, app.base, app.aa = store, base, aa          # exposed for tests
    return app

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>UtiliSave</title>
<style>
:root{--bg:#f6f8fa;--card:#fff;--ink:#14212b;--mut:#5b6b78;--acc:#0a7d5c;--warn:#a15c00;--line:#dde4ea}
@media(prefers-color-scheme:dark){:root{--bg:#0e161d;--card:#16222c;--ink:#e8f0f6;--mut:#9db0bf;--acc:#2ec29a;--warn:#f0b347;--line:#26394a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,"Noto Sans Devanagari","Noto Sans Tamil","Noto Sans Telugu",sans-serif}
main{max-width:560px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.4rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:12px 0}
label{display:block;font-weight:600;margin:12px 0 4px}input,select,textarea{width:100%;padding:12px;font-size:1rem;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
button{width:100%;padding:14px;font-size:1.05rem;font-weight:700;border:0;border-radius:12px;background:var(--acc);color:#fff;margin-top:14px;cursor:pointer}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line)}button:disabled{opacity:.5}
.mut{color:var(--mut);font-size:.92rem}.warn{color:var(--warn);font-weight:600}.err{color:#c0392b;font-weight:600}
.chk{display:flex;gap:10px;align-items:flex-start;margin:10px 0}.chk input{width:22px;height:22px;margin-top:3px}
.tag{display:inline-block;padding:3px 10px;border-radius:99px;background:var(--acc);color:#fff;font-size:.85rem;font-weight:700}
li{margin:6px 0}details summary{cursor:pointer;font-weight:600}
</style></head><body><main>
<h1>UtiliSave</h1><noscript><p class="err">This page needs JavaScript. Please turn it on and refresh.</p></noscript><p class="mut" data-i="tag"></p>
<label for="lang" data-i="lang"></label>
<select id="lang"><option value="en">English</option><option value="hi">हिन्दी</option><option value="ta">தமிழ்</option><option value="te">తెలుగు</option></select>
<div id="msg" role="status" aria-live="polite"></div>
<section id="inbox" class="card" hidden><b id="inbox_t" data-i="inbox"></b><div id="inbox_b"></div></section>

<section id="s1" class="card"><b data-i="s1"></b>
<label for="mobile" data-i="mobile"></label><input id="mobile" inputmode="numeric" maxlength="10" autocomplete="tel">
<div id="otpbox" hidden><label for="otp" data-i="code"></label><input id="otp" inputmode="numeric" maxlength="6">
<p class="mut" id="otphint"></p></div><button id="b1"></button></section>

<section id="s2" class="card" hidden><b data-i="s2"></b>
<label for="vpa" data-i="upi"></label><input id="vpa" placeholder="name@bank" autocomplete="off">
<p class="mut" data-i="upi_help"></p>
<div id="aabox" hidden><label for="aaid" data-i="aa_id"></label><input id="aaid" placeholder="9876543210@provider" autocomplete="off"><p class="mut" data-i="aa_help"></p></div>
<label for="occ" data-i="occ"></label><select id="occ"></select>
<details open><summary data-i="notice_title"></summary><div id="notice" class="mut"></div></details>
<div class="chk"><input type="checkbox" id="adult"><label for="adult" style="margin:0;font-weight:400" data-i="adult"></label></div>
<div class="chk"><input type="checkbox" id="consent"><label for="consent" style="margin:0;font-weight:400" data-i="consent"></label></div>
<button id="b2" data-i="check"></button></section>

<section id="s3" class="card" hidden><b data-i="working"></b><ol id="steps" class="mut"></ol><button id="cont" data-i="cont" hidden></button></section>

<section id="s4" hidden>
<div class="card"><span class="tag" id="lvl"></span><h2 id="head" style="margin:.5rem 0"></h2><p id="script"></p><p id="demo" class="warn"></p></div>
<div class="card"><b data-i="next"></b><ul id="next"></ul></div>
<div class="card"><b data-i="why"></b><ul id="why"></ul><b id="impt" data-i="improve"></b><ul id="imp"></ul></div>
<div class="card mut" id="disc"></div>
<button class="sec" id="contest" data-i="contest"></button>
<button class="sec" id="correct" data-i="correct"></button>
<div id="corrbox" class="card" hidden><label for="occ2" data-i="occ"></label><select id="occ2"></select><button id="sendcorr" data-i="send_corr"></button></div>
<button class="sec" id="withdraw" data-i="withdraw"></button></section>
</main>
<script nonce="__NONCE__">
const $=id=>document.getElementById(id);let T=null,D=null,S={},L='en';
const say=(t,c)=>{$('msg').className=c||'';$('msg').textContent=t||''};
const li=t=>{const e=document.createElement('li');e.textContent=t;return e};
async function api(p,m,b){const r=await fetch(p,{method:m||'GET',headers:Object.assign({'Content-Type':'application/json'},T?{Authorization:'Bearer '+T}:{}),body:b?JSON.stringify(b):undefined});
 let j={};try{j=await r.json()}catch(e){}if(!r.ok)throw new Error(j.error||'Something went wrong');return j}
function renderNotice(n){const d=$('notice');d.textContent='';
 [[S.n_use,n.what_we_use.join(', ')+' - '+n.period],[S.n_why,n.why],[S.n_how,n.how_you_share],[S.n_long,n.how_long],[S.n_share,n.sharing],[S.n_auto,n.automated],[S.n_rights,n.your_rights.join('; ')],[S.n_comp,n.grievance_contact]]
 .forEach(([k,v])=>{const p=document.createElement('p');const b=document.createElement('b');b.textContent=k+': ';p.appendChild(b);p.appendChild(document.createTextNode(v));d.appendChild(p)})}
async function setLang(l){L=l;$('lang').value=l;const j=await (await fetch('/api/i18n/'+l)).json();S=j.ui;document.documentElement.lang=l;
 document.querySelectorAll('[data-i]').forEach(e=>{e.textContent=S[e.dataset.i]});
 const sel=$('occ'),cur=sel.value;sel.textContent='';Object.entries(j.occupations).forEach(([k,v])=>{const o=document.createElement('option');o.value=k;o.textContent=v;sel.appendChild(o)});if(cur)sel.value=cur;
 const n=await (await fetch('/api/notice?lang='+l)).json();window.NV=n.version;renderNotice(n);
 $('b1').textContent=$('otpbox').hidden?S.send:S.verify}
async function loadInbox(){try{const ns=(await api('/api/my/notifications')).filter(n=>!n.read);const box=$('inbox_b');box.textContent='';
 if(!ns.length){$('inbox').hidden=true;return}
 ns.forEach(n=>{const w=document.createElement('div');Object.entries(n.fields).forEach(([k,v])=>{const p=document.createElement('p');const b=document.createElement('b');b.textContent=n.labels[k]+': ';p.appendChild(b);p.appendChild(document.createTextNode(v));w.appendChild(p)});
  const btn=document.createElement('button');btn.textContent=S.ok_read;btn.onclick=async()=>{await api('/api/my/notifications/'+n.id+'/read','POST');loadInbox()};w.appendChild(btn);box.appendChild(w)});
 $('inbox_t').textContent=ns[0].title;$('inbox').hidden=false}catch(e){}}
$('lang').onchange=()=>setLang($('lang').value);
fetch('/healthz').then(r=>r.json()).then(h=>{$('aabox').hidden=!h.needs_aa_id});
const nl=(navigator.language||'en').slice(0,2);setLang(['hi','ta','te'].includes(nl)?nl:'en').catch(()=>say('Could not load the page text. Please refresh the page.','err'));
$('b1').onclick=async()=>{try{say('');const m=$('mobile').value.trim();
 if($('otpbox').hidden){const r=await api('/api/auth/otp/request','POST',{mobile:m});$('otpbox').hidden=false;$('b1').textContent=S.verify;
  if(r.sandbox_otp)$('otphint').textContent=S.demo_code+r.sandbox_otp}
 else{const r=await api('/api/auth/otp/verify','POST',{mobile:m,otp:$('otp').value.trim()});T=r.token;$('s1').hidden=true;$('s2').hidden=false;loadInbox()}}catch(e){say(e.message,'err')}};
$('b2').onclick=async()=>{try{say('');$('b2').disabled=true;$('s2').hidden=true;$('s3').hidden=false;const st=$('steps');st.textContent='';
 [S.w1,S.w2,S.w3,S.w4].forEach(t=>st.appendChild(li(t)));
 D=await api('/api/analyse','POST',{vpa:$('vpa').value.trim(),aa_id:$('aaid').value.trim(),occupation:$('occ').value,lang:L,adult:$('adult').checked,consent:$('consent').checked,notice_version:window.NV});
 if(D.status==='waiting_for_aa_approval'){say(D.message);$('cont').hidden=false;window.CID=D.consent_id}else show(D)}
 catch(e){$('s3').hidden=true;$('s2').hidden=false;$('b2').disabled=false;say(e.message,'err')}};
function show(d){$('s3').hidden=true;$('s4').hidden=false;$('lvl').textContent=S.level+d.level;$('head').textContent=d.headline;
 $('script').textContent=d.script||'';$('demo').textContent=d.demo_notice||'';
 const fill=(id,a,f)=>{const u=$(id);u.textContent='';(a||[]).forEach(x=>u.appendChild(li(f(x))))};
 fill('next',d.next_steps,x=>x.action+': '+x.name);fill('why',d.why,x=>x.text+' ('+x.effect+')');fill('imp',d.how_to_improve,x=>x);
 $('impt').hidden=!(d.how_to_improve||[]).length;$('disc').textContent=(d.data_note?d.data_note+' ':'')+d.disclosure+' '+S.kept+d.expires_at.slice(0,10)}
$('cont').onclick=async()=>{try{say('');D=await api('/api/analyse/'+window.CID+'/continue','POST');
 if(D.status==='waiting_for_aa_approval'){say(D.message)}else{$('cont').hidden=true;show(D)}}catch(e){say(e.message,'err')}};
$('correct').onclick=()=>{const o=$('occ2');o.textContent='';[...$('occ').options].forEach(x=>o.appendChild(x.cloneNode(true)));$('corrbox').hidden=false};
$('sendcorr').onclick=async()=>{try{await api('/api/my/decisions/'+D.id+'/correction','POST',{field:'occupation',value:$('occ2').value});$('corrbox').hidden=true;say(S.corr_done)}catch(e){say(e.message,'err')}};
$('contest').onclick=async()=>{try{const r=await api('/api/my/decisions/'+D.id+'/contest','POST',{reason:prompt(S.contest_prompt)||''});say(r.message)}catch(e){say(e.message,'err')}};
$('withdraw').onclick=async()=>{if(!confirm(S.confirm_w))return;try{const me=await api('/api/my/data');const c=me.consents.find(x=>x.status==='ACTIVE');
 if(c)await api('/api/my/consent/'+c.id+'/revoke','POST');await api('/api/my/data','DELETE');$('s4').hidden=true;say(S.done_w)}catch(e){say(e.message,'err')}};
</script></body></html>"""

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
