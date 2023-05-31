
from django.urls import path
from .views import home_page, register_user, login_page, logout_user, delete_task, update_task

urlpatterns = [
    path("", home_page, name="home-page"),
    path("register-user/", register_user, name="register-user"),
    path("login/", login_page, name="login-page"),
    path("logout/", logout_user, name="logout"),
    path("delete-task/<str:task>", delete_task, name="delete-task"),
    path("update-task/<str:task>", update_task, name="update-task")
]


