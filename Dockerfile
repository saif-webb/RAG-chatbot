# Portable image for any container host (Cloud Run, Fly, Railway, a VPS).
#
# Sizing: sentence-transformers pulls PyTorch. The CPU-only wheel keeps this to
# roughly 1 GB rather than the several GB the default CUDA build costs. Budget
# at least 1 GB of RAM at runtime — 512 MB is not enough.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the model cache inside the image rather than in a home directory that
    # may not be writable.
    HF_HOME=/opt/models

WORKDIR /srv

# CPU-only PyTorch first, so the resolver in the next step does not pull the
# multi-gigabyte CUDA wheels as a transitive dependency.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Bake the embedding weights into the image. Without this the first request
# after every cold start would download ~90 MB, and on hosts with an ephemeral
# filesystem that happens on every restart.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBEDDING_MODEL}')" \
    && chmod -R a+rX /opt/models

COPY backend/  backend/
COPY frontend/ frontend/
COPY scripts/  scripts/

# Do not run as root.
RUN useradd --create-home --uid 10001 app && chown -R app:app /srv
USER app

WORKDIR /srv/backend

ENV PORT=8000
EXPOSE 8000

# start-period is generous: the model loads before the app serves traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/healthz').status==200 else 1)"

# Shell form so $PORT is expanded — every managed host assigns the port.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
