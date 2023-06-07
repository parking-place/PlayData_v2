from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

from .models import todo


@login_required(login_url="user/login/")
def home_page(request):
    if request.method == "POST":
        task = request.POST.get("task")
        if len(task):
            new_todo = todo(user=request.user, todo_name=task)
            new_todo.save()
    all_todos = todo.objects.filter(user=request.user)
    context = {"todos": all_todos}
    return render(request, "todoList/todo.html", context)


@login_required(redirect_field_name="user-login")
def delete_task(request, task):
    get_todo = todo.objects.get(user=request.user, todo_name=task)
    get_todo.delete()
    return redirect("home-page")


@login_required(redirect_field_name="user-login")
def update_task(request, task):
    get_todo = todo.objects.get(user=request.user, todo_name=task)
    get_todo.status = True
    get_todo.save()
    return redirect("home-page")
