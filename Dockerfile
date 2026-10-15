# Cognitive Browser Microservice (Reflex Core + Playwright + DeBERTa-v3 NLI)
# Lightweight, self-contained container with native GPU (NVIDIA CUDA) support and CPU fallback.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEVICE=cuda \
    PORT=8000 \
    SESSION_DIR=/data/browser_sessions

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    xvfb \
    ca-certificates \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Copy and install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install Playwright Chromium browser engine
RUN playwright install chromium

# 5. Copy source code and install package
COPY pyproject.toml /app/
COPY cognitive_browser/ /app/cognitive_browser/
COPY mcp_server.py server.py /app/
RUN pip install --no-cache-dir -e .

# Create directory for browser sessions
RUN mkdir -p /data/browser_sessions

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "cognitive_browser.server:app", "--host", "0.0.0.0", "--port", "8000"]
