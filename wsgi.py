"""WSGI entry point:  gunicorn -c gunicorn.conf.py wsgi:app"""
import os
from app import create_app
app = create_app(n_workers=int(os.getenv("N_WORKERS", "1500")))
