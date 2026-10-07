from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import static


def home_redirect(request):
    return render(request, "predictor/home_landing.html")


urlpatterns = [
    path("admin/", admin.site.urls),

    path("accounts/", include("accounts.urls")),

    path("", home_redirect, name="home_redirect"),

    path("predictor/", include("predictor.urls")),
]


if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )