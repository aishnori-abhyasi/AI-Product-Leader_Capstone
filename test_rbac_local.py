import time
from flask import Flask, jsonify
import rbac

# 1. Create a minimal Flask app using your exact rbac.py module
app = Flask(__name__)

@app.get("/admin-only")
@rbac.require("system:revoke_sessions")
def admin_route():
    return jsonify(ok=True, user=rbac.g.claims["sub"])

@app.get("/analyst-only")
@rbac.require("explain:read_case")
def analyst_route():
    return jsonify(ok=True)

client = app.test_client()

# 2. Test Password Hashing (scrypt)
hashed = rbac.hash_password("SuperSecret123")
assert rbac.check_password("SuperSecret123", hashed) is True
assert rbac.check_password("WrongPass", hashed) is False
print("[PASS] scrypt password hashing works")

# 3. Issue tokens for two different roles
analyst_token = rbac.issue_token(sub="analyst_1", role="risk_analyst")
compliance_token = rbac.issue_token(sub="officer_1", role="compliance_officer")

# 4. Test Role Permissions (403 Forbidden vs 200 OK)
r1 = client.get("/admin-only", headers={"Authorization": f"Bearer {analyst_token}"})
assert r1.status_code == 403, f"Expected 403, got {r1.status_code}"

r2 = client.get("/admin-only", headers={"Authorization": f"Bearer {compliance_token}"})
assert r2.status_code == 200, f"Expected 200, got {r2.status_code}"
print("[PASS] RBAC permission boundaries enforced (403 vs 200)")

# 5. Test the Global Kill Switch (revoke_all_sessions)
time.sleep(0.01)  # Ensure 1ms passes so NOT_BEFORE_MS > token's 'ims'
rbac.revoke_all_sessions()

r3 = client.get("/admin-only", headers={"Authorization": f"Bearer {compliance_token}"})
assert r3.status_code == 401, f"Expected 401 after kill switch, got {r3.status_code}"
print("[PASS] Global session kill switch (NOT_BEFORE_MS) invalidated all active tokens!")