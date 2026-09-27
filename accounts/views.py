from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import UserProfile


def register(request):

    if request.method == "POST":

        name = request.POST.get("name", "").strip()
        gender = request.POST.get("gender", "").strip()
        email = request.POST.get("email", "").strip()
        phone = request.POST.get("phone", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        # Check all fields
        if not name or not gender or not email or not phone or not password:
            return render(
                request,
                "accounts/register.html",
                {"error": "Please fill all fields."}
            )

        # Check password
        if password != confirm_password:
            return render(
                request,
                "accounts/register.html",
                {"error": "Passwords do not match."}
            )

        # Check existing email
        if User.objects.filter(email__iexact=email).exists():
            return render(
                request,
                "accounts/register.html",
                {"error": "Email already registered."}
            )

        # Use email as internal username
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=name
        )

        # Save additional information
        UserProfile.objects.create(
            user=user,
            gender=gender,
            phone=phone
        )

        return redirect("login")

    return render(request, "accounts/register.html")


def login_view(request):

    if request.method == "POST":

        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")

        try:
            user_obj = User.objects.get(email__iexact=email)

            user = authenticate(
                request,
                username=user_obj.username,
                password=password
            )

        except User.DoesNotExist:
            user = None

        if user is not None:

            login(request, user)

            return redirect("home")

        return render(
            request,
            "accounts/login.html",
            {"error": "Invalid email or password."}
        )

    return render(request, "accounts/login.html")


def logout_view(request):

    logout(request)

    return redirect("login")