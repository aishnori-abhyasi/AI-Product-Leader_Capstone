# UtiliSave – gig-worker scheme navigator (MVP)

Customer enters a UPI ID → consent → Account Aggregator data (sandbox: synthetic) → 4 agents → plain-English suggestion.
**Status: sandbox MVP on synthetic data. Not legal advice. Not ready for real customers** (see "Go-live checklist" and "Known gaps").
Update 2: production-shaped AA adapter, Hindi/Tamil/Telugu, data-correction and breach-incident flows added (sections at the end).

```
pip install -r requirements.txt
python app.py                  # http://127.0.0.1:5000   demo OTP is shown on screen
python run_tests.py            # all six suites: agents, AA adapter, AA<->app, languages, corrections/incidents, security/fairness
ANTHROPIC_API_KEY=... python app.py     # turns the LangChain agents on (explanations + script rewriting)
ENV=prod APP_JWT_SECRET=.. APP_PEPPER=.. STAFF_DEMO_PASSWORD=.. python app.py   # refuses to start on defaults
```

## Three facts that shaped the design
1. **A UPI ID cannot fetch anyone's transactions** and anyone can type anyone's UPI ID. It is used only as a label; data
   arrives only via an Account Aggregator (AA) consent the real account holder approves in their AA app.
2. **The data user (FIU) in the AA framework must itself be regulated** (RBI/SEBI/IRDAI/PFRDA). A start-up needs to be one
   or work under a regulated partner. Offering loans additionally puts you in the RBI Digital Lending Directions 2025
   (regulated entity + lending service provider arrangement). The MVP therefore **recommends schemes; it does not lend**.
3. **UPI always sits on a bank account**, so "gig workers without bank entries" really means *thin credit file*, not
   *no bank account*. The no-bank-account rule is kept as a safe default.

## Architecture
```
Browser (mobile web) ─ OTP login ─▶ /api/analyse ─▶ consent ledger ─▶ AA gateway (sandbox)
                                                         │ telemetry held in memory only
                          gatekeeper ▶ Segmentation(K-Means) ▶ Scoring(Ridge LR) ▶ Matching(policy+RAG) ▶ Communication(Flesch-Kincaid RAG)
                                                         │ derived result only (no raw transactions) → 30-day retention
Staff API (RBAC) ◀─ audit (hash-chained) · SHAP · fairness report · model card · review queue
```
Explainability: exact linear **SHAP** (additivity tested; cross-checks against the `shap` library when installed);
customers see plain-English reasons + "how to improve"; analysts see SHAP values.

## RBAC (separation of duties)
| Capability | customer | support | risk_analyst | compliance_officer | admin |
|---|---|---|---|---|---|
| Own suggestion, contest, export, erase, withdraw | ✔ | | | | |
| Masked case list / review (four-eyes) | | ✔ | ✔ | | |
| SHAP + model internals per case | | | ✔ | | |
| Model card, fairness report | | | ✔ | ✔ | |
| Audit trail, consent ledger, retention purge | | | | ✔ | |
| Grievances | | ✔ | | ✔ | |
| Manage staff users | | | | | ✔ |
| Correct own occupation; read breach notices | ✔ | | | | |
| Process correction requests | | ✔ | | | |
| Declare an incident | | ✔ | ✔ | ✔ | ✔ |
| Read/manage incidents, notify people, Board-report draft | | | | ✔ | |
| Pause processing (kill switch), revoke all sessions | | | | ✔ | ✔ |
Admin cannot read customer data; compliance cannot read scores; support cannot see model internals. Denials are audited.

## Control map (verify against primary texts; secondary sources used)
**DPDP Act 2023 / Rules 2025** – Rules notified 13–14 Nov 2025. Rules 1, 2, 17–21 apply now; Rule 4 (consent managers) one
year after; the substantive duties (notice, consent, security, breach, rights – Rules 3, 5–16, 22–23) from about **13–14 May 2027**.
The older SPDI Rules continue until then. Build to the 2027 standard now.
| Requirement | Where | Status |
|---|---|---|
| Itemised plain-language notice, version-stamped | `compliance.notice()`, UI | Done – needs legal review + translations |
| Specific, informed consent; withdrawal as easy as giving | consent ledger, `/consent/<id>/revoke`, UI button | Done |
| Purpose limitation / data minimisation | single `PURPOSE`; raw transactions never stored | Done |
| Storage limitation | 30-day expiry, `purge_expired` | Done – **schedule the purge** |
| Access, correction, erasure, grievance | `/my/data`, `/my/decisions/<id>/correction`, `/my/grievance` | Done; correction covers **occupation only** (other data is derived or lives at the bank) |
| Security safeguards, logs | RBAC, pseudonymisation, tamper-evident audit | Partly – **TLS, encryption at rest, KMS, MFA, pen-test are infra tasks** |
| Breach notification (Board + affected people) | `incidents.py`, `/staff/incidents/*` | Workflow built (clocks, notices, Board-report draft, closing gates); **no phone/e-mail channel, so notices are in-app only** |
| Children (<18): verifiable parental consent | 18+ self-declaration only | **Weak** – decide: block minors robustly |
| Significant Data Fiduciary duties (DPO, DPIA, audit) | – | Assess whether you are designated |
**Account Aggregator** – purpose-specific consent artefact ("one purpose, one consent"), data life, revocation, encrypted
fetch (FIP encrypts to the FIU's key; the AA cannot read it). `SandboxAA` simulates it; `ProductionAA` lists the real call
sequence and is **not implemented**. Join Sahamati's ecosystem and follow its Fair Use templates for lending journeys.
**RBI Digital Lending Directions 2025** (apply to REs and their LSPs) – explicit, need-based consent with audit trail ✔; data
stored in India (**deploy in an Indian region; check what you send to any LLM API**); KFS/APR disclosure and DLA reporting by the
RE apply once you show loan offers (MVP does not).
**RBI FREE-AI report (Aug 2025, recommendations)** – AI disclosure to customers ✔, ability to contest ✔, fairness audits ✔
(below), model card ✔, human oversight ✔; **AI incident reporting process missing**.

## Fairness evidence (what the report is and is not)
`fairness.py` runs on held-out data: selection-rate parity (difference and four-fifths ratio), TPR/FPR and equalized-odds gaps,
AUC gap, calibration by group, and a proxy-feature check. A breach counts as **clear** only if it crosses the threshold **and**
is larger than chance (permutation test, p<0.05); otherwise it is **inconclusive**. Fairlearn is used and cross-checked when
installed (`pip install fairlearn`); native equivalents run otherwise. Sensitive attributes are used **only** for auditing, never
as model inputs, and need their own consent/purpose to collect.
**In this repo the groups and the disparities are synthetic** – it proves the machinery catches a real gap (gender/region flagged,
independent age/language not). It is *not* evidence that any real model is fair. There is no multiplicity correction (5 attributes × 3 metrics).
Mitigation order: fix proxies/policy first (e.g. graduated products for irregular earners), then Fairlearn `ExponentiatedGradient`;
never per-group thresholds at decision time.

## Go-live checklist
1. Appoint a regulated partner (FIU/RE) and sign the AA + LSP arrangements; get legal sign-off on notice, retention, DPO.
2. Implement `ProductionAA`; store keys in an HSM/KMS; Indian-region hosting; TLS; encryption at rest.
3. Replace synthetic data with real consented data; retrain; rerun `fairness.py` with real, consented audit attributes; independent validation; sign the model card.
4. Verify **every** scheme fact against official sources (MUDRA, PMSBY, APY, Jan Dhan are unverified in the KB); `safe_to_ship` stays false until then.
5. Staff SSO + MFA, secrets rotation, rate limiting at the gateway, pen-test, backups, scheduled purge, breach/incident runbook.
6. Add Hindi/Tamil/Telugu with native review (Flesch-Kincaid is English-only), a comprehension test with real gig workers, and a correction-of-data flow.
7. Monitoring: drift, calibration and fairness on a schedule; archive each report.

## Sources consulted (secondary – read the primary texts)
DPDP Rules 2025 timeline: mondaq.com (AZB; "a closer look at the DPDP Rules 2025"), legal500.com, storyboard18.com ·
AA framework: sahamati.org.in Fair Use (Dec 2025), hyperverge.co AA guide, business-standard.com on Sahamati SRO ·
RBI Digital Lending Directions 2025: rbi.org.in (notification 8 May 2025), mondaq.com/ssrana.in summaries ·
RBI FREE-AI report (13 Aug 2025): mondaq.com (Khaitan & Co).

## Update 2 - what was added and how far it is verified
**1. Production-shaped Account Aggregator adapter** (`aa_production.py`, `aa_simulator.py`, tests `test_aa.py`, `test_aa_app.py`)
Consent request limited to DEPOSIT + TRANSACTIONS, ONETIME fetch, 90-day range, 30-day data life; detached-JWS (RFC 7797) signing of
every call and verification of every webhook; revocation in the customer's AA app deletes our derived results; bank statements are
decrypted and mapped in memory. **Tested only against my simulator, which encodes my reading of the public specs.** Before use:
validate in Sahamati UAT (paths, `ver`, headers, purpose codes, consent template); run Sahamati's Rahasya service for the key
exchange (real FIPs use a Weierstrass-style Curve25519 encoding; `LocalX25519Crypto` is test-only and `RahasyaCrypto`'s request
shapes are from memory); become/partner with a regulated FIU. **Train-serve skew:** features derived from real statements are
proxies (bill "on time" becomes "bills paid regularly", payer identity comes from bank-specific narration text) - retrain on
features produced by `statement_to_telemetry()`. A missing bill signal is imputed neutrally and disclosed, never treated as late payment.

**2. Hindi, Tamil, Telugu** (`i18n.py`, `indic.py`, `test_i18n.py`)
Every customer-facing string, the DPDP notice, reasons, "how to improve", scheme facts/names and breach-notice labels. Scripts are
assembled from fixed translated fact templates so numbers cannot drift; tests assert every number, amount, URL and the
"street vendors only" caveat is identical to the English fact. Flesch-Kincaid is not used: `indic.py` reports sentence length,
akshara length and long-word share as ADVISORY, UNCALIBRATED signals. **All translations are AI-drafted. A localized script is
withheld in prod until `i18n.REVIEW[lang]` names a native-speaker reviewer, and the notice needs a legal translation.** Server error
messages are still English-only. A comprehension test with real gig workers is the only valid readability evidence.

**3. Data correction and breach/incident handling** (`incidents.py`, `/api/my/*correction*`, `/api/staff/incidents/*`, `test_incidents.py`)
Correction: customer requests a change of occupation; support applies it (the four agents re-run from stored derived features, no
new bank fetch) or rejects it; history kept. Incidents (my reading of DPDP Rule 7 - verify against the Gazette): affected people
are notified without delay in their notice language (English fallback flagged); Board initial intimation without delay and the
detailed report within 72 hours (extendable only by the Board); a 6-hour CERT-In clock where counsel marks it applicable; a Board-report
draft counting deliveries and reads; an incident cannot be closed until the required steps and a post-incident review are recorded;
kill switch (DPO/admin) and revoke-all-sessions. Every step is audited.

## Known gaps (not built)
- Contact channel for breach notices (phone/e-mail collected under a separate "security and legal notices" consent, stored encrypted).
- DPDP Rule 6 asks for one-year retention of logs: the audit table is never purged by this app, but confirm minimum retention in infra.
- Filing with the Data Protection Board is manual (no portal had surfaced in the sources I found).
- Correction of fields other than occupation; staff UI (staff use the API); staff MFA/SSO; encryption at rest, TLS, KMS, India hosting.
- Real AA integration, real data, real fairness attributes, independent model validation, native-speaker and legal review.
