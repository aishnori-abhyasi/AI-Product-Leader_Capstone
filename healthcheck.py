import os, sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{os.getenv('PORT', '8000')}/healthz", timeout=3); sys.exit(0)
except Exception:
    sys.exit(1)
