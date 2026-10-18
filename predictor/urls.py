
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="predictor_home"),

    path("predict/", views.predict, name="predict"),

    path("history/", views.history, name="history"),

    path(
        "history/delete-selected/",
        views.delete_selected_history,
        name="delete_selected_history"
    ),

    path(
        "history/delete/",
        views.delete_history,
        name="delete_history"
    ),
]