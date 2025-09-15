FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
EXPOSE 8000
USER appuser
CMD ["gunicorn", "-k", "uvicorn_worker.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
