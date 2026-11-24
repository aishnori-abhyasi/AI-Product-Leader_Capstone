"""Run every suite and print one line each. Usage: python run_tests.py"""
import subprocess, sys
SUITES = ["test_agents", "test_aa", "test_aa_app", "test_i18n", "test_incidents", "test_app"]
bad = 0
for s in SUITES:
    r = subprocess.run([sys.executable, s + ".py"], capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if "PASSED" in l]
    ok = r.returncode == 0 and bool(last)
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {s:<16}", last[-1] if last else (r.stderr.strip().splitlines() or ['?'])[-1])
sys.exit(1 if bad else 0)
