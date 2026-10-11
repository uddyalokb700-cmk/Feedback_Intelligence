"""
Development settings for Customer Feedback Intelligence (CFI).
"""

import os

from .base import *

DEBUG = True

ALLOWED_HOSTS = ["*"]

# Database configuration: PostgreSQL when configured, SQLite fallback for zero-dep local dev
DB_ENGINE = os.environ.get("DB_ENGINE", "")
DB_NAME = os.environ.get("DB_NAME", "")

if DB_ENGINE == "django.db.backends.postgresql" or (DB_NAME and not DB_NAME.endswith(".sqlite3") and os.environ.get("USE_POSTGRES") == "1"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("DB_NAME", "cfi_db"),
            "USER": os.environ.get("DB_USER", "postgres"),
            "PASSWORD": os.environ.get("DB_PASSWORD", "postgres"),
            "HOST": os.environ.get("DB_HOST", "localhost"),
            "PORT": os.environ.get("DB_PORT", "5432"),
            "CONN_MAX_AGE": 60,
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
            "CONN_MAX_AGE": 0,
        }
    }

# Ensure upload directory exists
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# Synchronous eager task execution by default in development unless explicit CELERY_TASK_ALWAYS_EAGER=False
CELERY_TASK_ALWAYS_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "True").lower() in ("true", "1")
