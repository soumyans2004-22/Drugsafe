from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="predictor_home"),
    path("predict/", views.predict, name="predict"),
    path("history/", views.history, name="history"),
]