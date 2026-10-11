from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    # Web UI authentication
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    # REST API authentication
    path("api/auth/register/", views.RegisterAPIView.as_view(), name="api_register"),
    path("api/auth/login/", views.LoginAPIView.as_view(), name="api_login"),
    path("api/auth/logout/", views.LogoutAPIView.as_view(), name="api_logout"),
    path("api/auth/me/", views.MeAPIView.as_view(), name="api_me"),
]
