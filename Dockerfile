# ── Stage: base image ─────────────────────────────────────────────────────────
# python:3.11-slim = official Python image, Debian-based, stripped of extras.
# "slim" shaves ~200MB vs the full image. Always pin the minor version so
# Render doesn't silently pull a different Python on the next deploy.
FROM python:3.11-slim

# ── Working directory ──────────────────────────────────────────────────────────
# Every subsequent COPY / RUN / CMD is relative to this path inside the container.
WORKDIR /app

# ── Dependencies (own layer, cached separately from code) ─────────────────────
# Copy requirements FIRST before any app code. Docker builds layers in order
# and caches each one. If requirements.txt hasn't changed, pip install is
# skipped on the next build even if your Python files changed — saves minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
# Copy only what the app actually needs at runtime. Src/ and data/ stay out.
COPY app/        ./app/
COPY templates/  ./templates/
COPY static/     ./static/

# ── Model artifacts ────────────────────────────────────────────────────────────
# Bake the _latest files directly into the image. No external storage needed.
# If the model is ever retrained, rebuild + redeploy the image.
COPY model/rf_model_latest.joblib        ./model/
COPY model/preprocessors_latest.joblib  ./model/

# ── Port ───────────────────────────────────────────────────────────────────────
# EXPOSE is documentation only — it doesn't open the port. Render reads this
# hint but ultimately you configure the port in the Render dashboard.
EXPOSE 8000

# ── Start command ──────────────────────────────────────────────────────────────
# CRITICAL: --host 0.0.0.0 binds to all interfaces inside the container.
# If you use 127.0.0.1 (the default), the app is only reachable from inside
# the container itself — Render's proxy can't reach it and the deploy fails.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
