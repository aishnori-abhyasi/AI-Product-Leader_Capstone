import os
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"     # Cloud Run / App Runner inject PORT
# ONE worker on purpose: OTPs, pending consents, rate limits, token revocations and the sandbox AA live in process memory.
# More workers or instances would break sign-in. Scale only after moving that state to Redis/Postgres.
workers, threads = 1, int(os.getenv("THREADS", "8"))
preload_app = True          # train the models once, before serving
timeout = 120               # slow CPUs need time to train at start-up
accesslog, errorlog = "-", "-"
