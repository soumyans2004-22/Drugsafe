from django.contrib import admin
from django.urls import path, include

from django.shortcuts import render


def home_redirect(request):
    return render(request, "predictor/home_landing.html")

urlpatterns = [
    path("admin/", admin.site.urls),

    path("accounts/", include("accounts.urls")),

    path("", home_redirect, name="home_redirect"),

    path("predictor/", include("predictor.urls")),
]