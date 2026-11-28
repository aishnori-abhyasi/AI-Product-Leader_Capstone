FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt
COPY . .
RUN useradd --system --uid 10001 app && mkdir /data && chown app /data
USER 10001
# No secrets here. Provide APP_JWT_SECRET, APP_PEPPER, STAFF_DEMO_PASSWORD (and ANTHROPIC_API_KEY if wanted) at run time.
ENV ENV=sandbox APP_DB=/data/utilisave.db PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD ["python", "healthcheck.py"]
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
