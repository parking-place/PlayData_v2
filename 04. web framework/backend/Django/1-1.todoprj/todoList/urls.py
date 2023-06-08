from django.urls import path
from .views import home_page, delete_task, update_task

urlpatterns = [
    path("", home_page, name="home-page"),
    path("delete-task/<str:task>", delete_task, name="delete-task"),
    path("update-task/<str:task>", update_task, name="update-task"),
]
