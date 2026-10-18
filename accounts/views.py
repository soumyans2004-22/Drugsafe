from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST

from .models import UserProfile


def register(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        gender = request.POST.get("gender", "").strip()
        email = request.POST.get("email", "").strip()
        phone = request.POST.get("phone", "").strip()

        professional_role = request.POST.get(
            "professional_role", ""
        ).strip()

        registration_number = request.POST.get(
            "registration_number", ""
        ).strip()

        registration_proof = request.FILES.get("registration_proof")
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        def error(message):
            return render(request, "accounts/register.html", {
                "error": message
            })

        # Check required fields
        if not all([
            name, gender, email, phone, professional_role,
            registration_number, password, confirm_password
        ]) or not registration_proof:
            return error(
                "Please complete all fields and upload your proof."
            )

        if gender not in ["Male", "Female", "Other"]:
            return error("Please select a valid gender.")

        if professional_role not in [
            "Doctor", "Pharmacist", "Medical Professional"
        ]:
            return error("Please select a valid professional role.")

        try:
            validate_email(email)
        except ValidationError:
            return error("Enter a valid email address.")

        if User.objects.filter(email__iexact=email).exists():
            return error("Email already registered.")

        if password != confirm_password:
            return error("Passwords do not match.")

        if len(password) < 8:
            return error(
                "Password must contain at least 8 characters."
            )

        # Validate registration proof
        filename = registration_proof.name.lower()
        allowed_extensions = (".pdf", ".jpg", ".jpeg", ".png")

        if not filename.endswith(allowed_extensions):
            return error(
                "Upload a PDF, JPG, JPEG, or PNG document."
            )

        if registration_proof.size > 5 * 1024 * 1024:
            return error(
                "The proof document must be 5 MB or smaller."
            )

        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=email,
                    email=email,
                    password=password,
                    first_name=name
                )

                UserProfile.objects.create(
                    user=user,
                    gender=gender,
                    phone=phone,
                    professional_role=professional_role,
                    registration_number=registration_number,
                    registration_proof=registration_proof,
                    verification_status="Pending"
                )

        except Exception:
            return error(
                "Registration could not be completed. "
                "Please check your details and try again."
            )

        return render(request, "accounts/login.html", {
            "success": (
                "Registration submitted successfully. "
                "Your professional credentials are pending "
                "administrator review."
            )
        })

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

        if user is None:
            return render(request, "accounts/login.html", {
                "error": "Invalid email or password."
            })

        try:
            profile_data = user.profile
        except UserProfile.DoesNotExist:
            return render(request, "accounts/login.html", {
                "error": (
                    "Professional profile not found. "
                    "Contact the administrator."
                )
            })

        if profile_data.verification_status != "Approved":
            if profile_data.verification_status == "Pending":
                message = (
                    "Your credentials are awaiting administrator approval."
                )
            else:
                message = (
                    "Your verification was rejected. "
                    "Contact the administrator."
                )

            return render(request, "accounts/login.html", {
                "error": message
            })

        login(request, user)
        return redirect("predictor_home")

    return render(request, "accounts/login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required
def profile(request):
    try:
        profile_data = request.user.profile
    except UserProfile.DoesNotExist:
        return HttpResponseForbidden(
            "Professional profile not found."
        )

    return render(request, "accounts/profile.html", {
        "profile_data": profile_data
    })