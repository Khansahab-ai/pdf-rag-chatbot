FROM python:3.12-slim

# Create a non-root user (Hugging Face Spaces runs as UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch to keep image lightweight and build fast
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, data, Chroma DB and frontend UI
COPY --chown=user:user src ./src
COPY --chown=user:user api ./api
COPY --chown=user:user chroma_db ./chroma_db
COPY --chown=user:user data ./data
COPY --chown=user:user frontend ./frontend

# Ensure cache directory permissions for model downloads
RUN mkdir -p /home/user/.cache && chown -R user:user /home/user/.cache

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]