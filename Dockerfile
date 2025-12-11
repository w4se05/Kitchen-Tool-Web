FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    ffmpeg \
    libopus0 \
    libvpx7 \
    libssl-dev libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip wheel setuptools

RUN --mount=type=cache,target=/root/.cache/pip \
    test -f requirements.txt && /app/.venv/bin/pip install -r requirements.txt || true

COPY . /app

FROM python:3.12-slim-bookworm AS runtime

LABEL maintainer="Chau Minh Quan (CallMeQan)"
LABEL maintainer.email="minhquan99k@gmail.com"
LABEL maintainer.school="Vietnamese-German University"
LABEL maintainer.linkedin="chau-minh-quan"
LABEL image.description="Secure, minimal Streamlit app"

ARG APP_USER=appuser
RUN useradd -m -u 1000 ${APP_USER}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    ffmpeg \
    libopus0 \
    libvpx7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLCONFIGDIR=/tmp/matplotlib \
    # Streamlit settings
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Pre-create runtime writable dirs for non-root
RUN mkdir -p /tmp/matplotlib && chown -R ${APP_USER}:${APP_USER} /tmp/matplotlib /app

USER ${APP_USER}

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "Menu.py", "--server.port=${STREAMLIT_SERVER_PORT}", "--server.headless=${STREAMLIT_SERVER_HEADLESS}"]