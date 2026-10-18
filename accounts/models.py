from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):

    GENDER_CHOICES = [
        ("Male", "Male"),
        ("Female", "Female"),
        ("Other", "Other"),
    ]

    PROFESSIONAL_CHOICES = [
        ("Doctor", "Doctor"),
        ("Pharmacist", "Pharmacist"),
        ("Medical Professional", "Medical Professional"),
    ]

    VERIFICATION_CHOICES = [
        ("Pending", "Pending"),
        ("Approved", "Approved"),
        ("Rejected", "Rejected"),
    ]

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile"
    )

    gender = models.CharField(
        max_length=10,
        choices=GENDER_CHOICES
    )

    phone = models.CharField(max_length=20)

    professional_role = models.CharField(
        max_length=30,
        choices=PROFESSIONAL_CHOICES,
        default="Medical Professional"
    )

    registration_number = models.CharField(
        max_length=100,
        blank=True
    )

    registration_proof = models.FileField(
        upload_to="registration_proofs/",
        blank=True,
        null=True
    )

    verification_status = models.CharField(
        max_length=10,
        choices=VERIFICATION_CHOICES,
        default="Pending"
    )

    def __str__(self):
        return f"{self.user.username} - {self.verification_status}"