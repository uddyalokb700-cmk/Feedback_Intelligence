"""
URL configuration for Customer Feedback Intelligence (CFI).
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def health_check(request):
    """Safe application health endpoint."""
    return JsonResponse(
        {
            "status": "healthy",
            "service": "customer-feedback-intelligence",
            "version": "1.0.0",
        }
    )


urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", health_check, name="health-check"),
    path("", include("accounts.urls")),
    path("", include("datasets.urls")),
    path("", include("reviews.urls")),
    path("", include("analytics.urls")),
    path("", include("sentiment.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
