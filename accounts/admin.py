from django.contrib import admin
from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "professional_role",
        "registration_number",
        "verification_status",
    )

    list_filter = (
        "professional_role",
        "verification_status",
    )

    search_fields = (
        "user__username",
        "user__email",
        "user__first_name",
        "registration_number",
    )

    list_editable = ("verification_status",)

    readonly_fields = ("user",)

    fields = (
        "user",
        "gender",
        "phone",
        "professional_role",
        "registration_number",
        "registration_proof",
        "verification_status",
    )