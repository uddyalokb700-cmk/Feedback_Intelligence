# Customer Feedback Intelligence (CFI) - Production Dockerfile
FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr and writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (curl for health check, libpq for PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy project files
COPY . /app/

# Create non-root user and grant permissions
RUN useradd -m -u 1000 cfi && \
    mkdir -p /app/media/uploads /app/staticfiles /app/ml_artifacts && \
    chown -R cfi:cfi /app

USER cfi

# Collect static files during build
ENV DJANGO_SETTINGS_MODULE=config.settings.production
ENV DJANGO_SECRET_KEY=docker-build-temporary-secret-key-1234567890

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--threads", "2"]
