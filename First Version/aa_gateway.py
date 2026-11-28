"""
Account Aggregator (AA) gateway.

IMPORTANT FACTS THIS MODULE IS BUILT AROUND
  * A UPI ID (VPA) is only a payment address. It cannot be used to pull anyone's transaction history,
    and anyone can type anyone else's UPI ID. Data can only arrive through a consent flow the real
    account holder approves in their AA app.  So the UPI ID is just a label here.
  * Under the RBI NBFC-AA framework the data user (FIU) must itself be a regulated entity
    (bank / NBFC / insurer / etc.). A start-up either becomes one or operates under a regulated
    partner (and, for lending, an RE-LSP arrangement under the RBI Digital Lending Directions 2025).
  * Consent artefacts are purpose-specific ("one purpose, one consent"), time-bound, revocable and
    carry a data-life limit. Data arrives encrypted for the FIU's key; the AA cannot read it.

SandboxAA simulates that flow with synthetic data so the product can be built and tested.
ProductionAA documents the real call sequence; it is deliberately not implemented here.
"""
import uuid

from upi_agents_mvp import generate_upi_telemetry

class SandboxAA:
    def __init__(self): self._c = {}

    def request_consent(self, subject, purpose, fi_days, data_life_days, hint="", aa_id=None):
        handle = "aa-" + uuid.uuid4().hex[:12]
        self._c[handle] = {"subject": subject, "purpose": purpose, "fi_days": fi_days,
                           "data_life_days": data_life_days, "hint": hint.lower(), "status": "PENDING"}
        return handle

    def approve(self, handle):                    # in reality the customer taps "approve" inside their AA app
        self._c[handle]["status"] = "APPROVED"

    def revoke(self, handle):
        if handle in self._c: self._c[handle]["status"] = "REVOKED"

    def status(self, handle): return self._c.get(handle, {}).get("status", "UNKNOWN")

    def fetch(self, handle):
        """Returns a worker-day telemetry frame. Held in memory only; callers must not persist it."""
        c = self._c.get(handle)
        if not c or c["status"] != "APPROVED":
            raise PermissionError("consent not approved")
        arch = next((a for a in ("steady", "irregular", "mixed") if a in c["hint"]), None)   # demo hint
        seed = int(c["subject"][:8], 16) % (2 ** 31)
        tel, _ = generate_upi_telemetry(n_workers=1, days=c["fi_days"], seed=seed, force_archetype=arch)
        tel["worker_id"] = c["subject"]
        return tel

class ProductionAA:
    """Sequence to implement with a licensed AA (Finvu, OneMoney, CAMS Finserv, ...) per the ReBIT / Sahamati
    technical specs. Do not treat this list as the spec - use the official documents.
      1. FIU -> AA   POST /Consent         purpose code, FI types (DEPOSIT), FIDataRange (last 90 days),
                                          DataLife, fetch type (ONETIME), consent mode (VIEW)
      2. Customer approves in the AA app (mobile-number OTP) -> signed consent artefact
      3. AA  -> FIU  consent notification;  FIU -> AA  POST /Consent/fetch (get the artefact)
      4. FIU -> AA   POST /FI/request       (session id)
      5. AA relays to each FIP; FIP encrypts the data for the FIU's public key
      6. FIU -> AA   POST /FI/fetch ; FIU decrypts with its private key (keep keys in an HSM/KMS)
      7. Process in memory, store only derived results, honour DataLife and revocation (/Consent/revoke)
    """
    def __init__(self, *a, **k): raise NotImplementedError("integrate with your AA / regulated partner")
