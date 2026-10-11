"""
Production settings for Customer Feedback Intelligence (CFI).
"""

import os

from .base import *

DEBUG = False

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")
    if host.strip()
]

# Database configuration (PostgreSQL mandatory in production)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "cfi_db"),
        "USER": os.environ.get("DB_USER", "cfi_user"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "postgres"),
        "PORT": os.environ.get("DB_PORT", "5432"),
        "CONN_MAX_AGE": 600,
    }
}

# Real Celery queue in production
CELERY_TASK_ALWAYS_EAGER = False

# Security settings
SECURE_SSL_REDIRECT = os.environ.get("SECURE_SSL_REDIRECT", "True").lower() in ("true", "1")
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "True").lower() in ("true", "1")
CSRF_COOKIE_SECURE = os.environ.get("CSRF_COOKIE_SECURE", "True").lower() in ("true", "1")
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

HSTS_SECONDS = int(os.environ.get("SECURE_HSTS_SECONDS", 31536000))
if HSTS_SECONDS > 0:
    SECURE_HSTS_SECONDS = HSTS_SECONDS
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# Ensure media directory exists
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(ML_MODELS_DIR, exist_ok=True)
