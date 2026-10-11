"""
Views for authentication: HTML views and REST API endpoints.
"""

from django.contrib.auth import login, logout
from django.shortcuts import redirect, render
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import LoginSerializer, RegisterSerializer, UserSerializer


# HTML Template Views
def login_view(request):
    """Render the user login page."""
    if request.user.is_authenticated:
        return redirect("datasets:dashboard")
    return render(request, "registration/login.html")


def register_view(request):
    """Render the user registration page."""
    if request.user.is_authenticated:
        return redirect("datasets:dashboard")
    return render(request, "registration/register.html")


def logout_view(request):
    """Log the user out and redirect to login."""
    logout(request)
    return redirect("accounts:login")


# REST API Views
class RegisterAPIView(APIView):
    """
    POST /api/auth/register/
    Register a new user account with secure validation and automatic session creation.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            login(request, user)
            return Response(
                {
                    "message": "Registration successful.",
                    "user": UserSerializer(user).data,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginAPIView(APIView):
    """
    POST /api/auth/login/
    Authenticate user and create session.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            user = serializer.validated_data["user"]
            login(request, user)
            return Response(
                {
                    "message": "Login successful.",
                    "user": UserSerializer(user).data,
                },
                status=status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutAPIView(APIView):
    """
    POST /api/auth/logout/
    Terminate the authenticated user's session.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        logout(request)
        return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)


class MeAPIView(APIView):
    """
    GET /api/auth/me/
    Retrieve authenticated user profile and dataset counts.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        data = UserSerializer(user).data
        data["dataset_count"] = user.datasets.count()
        return Response(data, status=status.HTTP_200_OK)
