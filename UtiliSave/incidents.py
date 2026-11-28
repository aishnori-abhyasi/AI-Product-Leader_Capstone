"""
Incident and personal-data-breach workflow.

Built around my reading of DPDP Rule 7 (in force around 13 May 2027 - build to it now). VERIFY against the Gazette text:
  * affected Data Principals: notified WITHOUT DELAY, in plain language, with (a) what happened - nature, extent, timing;
    (b) likely consequences for them; (c) what the fiduciary is doing to mitigate; (d) safety steps they can take;
    (e) a contact person for questions.
  * Data Protection Board: initial intimation WITHOUT DELAY (nature, extent, timing, location, likely impact), then a
    detailed report within 72 HOURS of becoming aware (extendable only on the Board's written permission), including
    events/circumstances/reasons, mitigation, findings about the cause/person responsible, remedial measures to
    prevent recurrence, and a report on the intimations given to Data Principals.
  * no risk threshold: any personal data breach counts.
  * separate clocks may also run: CERT-In (6 hours for specified cyber incidents) and your regulated partner's / RBI's
    own incident-reporting duties. `cert_in_applicable` is a decision for counsel/CISO, recorded here.
Delivery limit: this app deliberately keeps no phone number/e-mail, so notices are delivered IN-APP (next login). To reach
people who do not open the app you need a contact channel collected under a separate "security and legal notices" consent
and stored encrypted. That is NOT built - see README gaps.
"""
import uuid
from datetime import datetime, timedelta, timezone

from compliance import iso, now

SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
KINDS = ("personal_data_breach", "cyber_incident", "availability_incident")
STATUSES = ("DECLARED", "CONTAINED", "NOTIFIED", "CLOSED")
EVENTS = ("contained", "board_initial", "board_report", "cert_in", "partner")
NOTICE_FIELDS = ("what", "when", "data", "effect", "doing", "you", "contact")      # Rule 7(1) elements
SLA_BOARD_INITIAL_H = 6          # INTERNAL target for "without delay"; not a statutory number

def _dt(s): return datetime.fromisoformat(s)

def validate_notice(notice):
    en = (notice or {}).get("en") or {}
    missing = [f for f in NOTICE_FIELDS if not str(en.get(f, "")).strip()]
    if missing: raise ValueError(f"English notice must fill: {missing}")
    for lang, fields in notice.items():
        for f, v in fields.items():
            if f not in NOTICE_FIELDS or len(str(v)) > 1500: raise ValueError(f"bad notice field {lang}.{f}")

def declare(store, actor, role, *, title, description, kind, personal_data_involved, severity, categories=None,
            scope=None, cert_in_applicable=False, aware_at=None, notice=None):
    if kind not in KINDS or severity not in SEVERITIES: raise ValueError("invalid kind or severity")
    if not str(title).strip() or not str(description).strip(): raise ValueError("title and description are required")
    t = now()
    aware = _dt(aware_at) if aware_at else t
    if aware.tzinfo is None: aware = aware.replace(tzinfo=timezone.utc)
    if aware > t + timedelta(minutes=1): raise ValueError("aware_at cannot be in the future")
    if notice: validate_notice(notice)
    inc = {"id": "i_" + uuid.uuid4().hex[:10], "title": str(title)[:200], "description": str(description)[:4000], "kind": kind,
           "severity": severity, "personal_data_involved": bool(personal_data_involved), "categories": list(categories or []),
           "scope": scope or {"all_active": True}, "cert_in_applicable": bool(cert_in_applicable), "aware_at": iso(aware),
           "declared_at": iso(t), "declared_by": actor, "status": "DECLARED", "events": [], "notice": notice or {}, "post_mortem": None}
    store.save_incident(inc)
    store.audit(actor, role, "incident_declared", inc["id"], severity=severity, kind=kind, personal_data=bool(personal_data_involved))
    return inc

def clocks(inc, at=None):
    t, aware = at or now(), _dt(inc["aware_at"])
    done = {e["event"]: e for e in inc["events"]}
    def row(key, hours):
        due = aware + timedelta(hours=hours); d = done.get(key)
        return {"due": iso(due), "done_at": d["at"] if d else None,
                "status": "DONE" if d else ("OVERDUE" if t > due else "PENDING"),
                "hours_left": None if d else round((due - t).total_seconds() / 3600, 1)}
    out = {"aware_at": inc["aware_at"], "as_of": iso(t)}
    if inc["personal_data_involved"]:
        out["board_initial (without delay; internal target %dh)" % SLA_BOARD_INITIAL_H] = row("board_initial", SLA_BOARD_INITIAL_H)
        out["board_report_72h (statutory; extendable only by the Board)"] = row("board_report", 72)
        out["principals"] = {"rule": "without delay", "status": "DONE" if inc.get("principals_notified_at") else "PENDING",
                             "done_at": inc.get("principals_notified_at")}
    out["cert_in_6h"] = row("cert_in", 6) if inc["cert_in_applicable"] else {"applicable": False}
    return out

def record_event(store, inc, event, actor, role, reference="", note=""):
    if event not in EVENTS: raise ValueError("unknown event")
    if any(e["event"] == event for e in inc["events"]): raise ValueError(f"{event} already recorded")
    inc["events"].append({"event": event, "at": iso(now()), "by": actor, "reference": str(reference)[:200], "note": str(note)[:1000]})
    if event == "contained" and inc["status"] == "DECLARED": inc["status"] = "CONTAINED"
    store.save_incident(inc); store.audit(actor, role, "incident_event", inc["id"], event=event)
    return inc

def notify_principals(store, inc, actor, role):
    """Create an in-app notice for every affected person, in the language of their consent (English fallback, flagged)."""
    if not inc["personal_data_involved"]: raise ValueError("no personal data involved: Rule 7 notices not applicable (document that assessment)")
    validate_notice(inc.get("notice"))
    created = fallback = skipped = 0
    for owner, lang in store.owners_in_scope(inc["scope"]):
        if store.has_notification(owner, inc["id"]): skipped += 1; continue
        fields = inc["notice"].get(lang); used = lang
        if not fields or any(not str(fields.get(f, "")).strip() for f in NOTICE_FIELDS):
            fields, used, fallback = inc["notice"]["en"], "en", fallback + 1
        store.add_notification(owner, inc["id"], {"incident": inc["id"], "severity": inc["severity"], "lang": used, "wanted_lang": lang,
                                                  "fields": {f: fields[f] for f in NOTICE_FIELDS}})
        created += 1
    inc["principals_notified_at"] = inc.get("principals_notified_at") or iso(now())
    if inc["status"] in ("DECLARED", "CONTAINED"): inc["status"] = "NOTIFIED"
    store.save_incident(inc)
    store.audit(actor, role, "principals_notified", inc["id"], created=created, language_fallback=fallback, already_notified=skipped)
    return {"created": created, "english_fallback": fallback, "already_notified": skipped}

def set_status(store, inc, new, actor, role, post_mortem=None):
    if new not in STATUSES: raise ValueError("invalid status")
    if new == "CLOSED":
        have = {e["event"] for e in inc["events"]}
        if inc["personal_data_involved"]:
            need = [x for x in ("board_initial", "board_report") if x not in have] + ([] if inc.get("principals_notified_at") else ["principals_notified"])
            if need: raise PermissionError(f"cannot close: missing {need}")
        if inc["cert_in_applicable"] and "cert_in" not in have: raise PermissionError("cannot close: missing ['cert_in']")
        if not str(post_mortem or "").strip(): raise PermissionError("cannot close: a post-incident review is required")
        inc["post_mortem"] = str(post_mortem)[:4000]
    inc["status"] = new
    store.save_incident(inc); store.audit(actor, role, "incident_status", inc["id"], status=new)
    return inc

def board_report(store, inc):
    """DRAFT for the legal/DPO team. Filing is done manually through the Board's mechanism (verify what it is at the time)."""
    stats = store.notification_stats(inc["id"]); ev = {e["event"]: e for e in inc["events"]}
    rep = {"status": "DRAFT - review with counsel before filing", "incident": inc["id"], "title": inc["title"],
           "initial_intimation (nature, extent, timing, location, likely impact)": {
               "nature": inc["description"], "categories_of_data": inc["categories"], "became_aware_at": inc["aware_at"],
               "scope": inc["scope"], "likely_impact": inc["notice"].get("en", {}).get("effect", "TO BE COMPLETED")},
           "detailed_report_72h": {
               "events_circumstances_reasons": "TO BE COMPLETED from investigation", "mitigation_measures": inc["notice"].get("en", {}).get("doing", "TO BE COMPLETED"),
               "findings_on_cause_and_responsible_person": "TO BE COMPLETED", "remedial_measures_to_prevent_recurrence": "TO BE COMPLETED",
               "intimations_to_data_principals": {**stats, "notified_at": inc.get("principals_notified_at"), "channel": "in-app only (see README gap)"},
               "contact_for_queries": inc["notice"].get("en", {}).get("contact", "TO BE COMPLETED")},
           "filings": {k: ev.get(k) for k in ("board_initial", "board_report", "cert_in", "partner")}}
    return rep

def public_view(inc, with_clocks=True):
    v = {k: inc[k] for k in ("id", "title", "kind", "severity", "status", "personal_data_involved", "cert_in_applicable",
                              "aware_at", "declared_at", "declared_by", "events", "categories", "scope", "principals_notified_at") if k in inc}
    if with_clocks: v["clocks"] = clocks(inc)
    return v
