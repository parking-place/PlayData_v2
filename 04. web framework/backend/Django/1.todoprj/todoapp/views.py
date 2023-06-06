from django.shortcuts import render, redirect
from django.contrib.auth.models import User 
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout 
from django.contrib import messages 

from .models import todo

@login_required(login_url="login/")
def home_page(request):
    if request.method == "POST":
        task = request.POST.get('task')
        if len(task):
            new_todo = todo(user=request.user, todo_name=task)
            new_todo.save()
    all_todos = todo.objects.filter(user=request.user)
    context = {
        'todos': all_todos
    }
    return render(request, 'todoapp/todo.html', context)

def register_user(request):
    if request.user.is_authenticated:
        return redirect('home-page')
    
    elif request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        pwd = request.POST.get('password')

        if len(pwd) < 3:
            messages.error(request, 'Password must be at least 3 characters')
            return redirect('register-user')
    
        get_all_users_by_username = User.objects.filter(username=username)
        if get_all_users_by_username:
            messages.error(request, 'Error, username already exists, User another.')
            return redirect('register-user')
        
        new_user = User.objects.create_user(username=username, email=email, password=pwd)
        new_user.save() 

        messages.success(request, 'User successfully created, login now')
        return redirect('login-page')

    else:
        return render(request, 'todoapp/register.html', {})

def login_page(request):
    if request.user.is_authenticated:
        return redirect('home-page')
    elif request.method == 'POST':
        username = request.POST.get('uname')
        pwd = request.POST.get('pass')

        validate_user = authenticate(username=username, password=pwd)
        if validate_user is not None:
            login(request, validate_user)
            return redirect('home-page')
        else:
            messages.error(request, 'Erro, wrong user details or user does not exist')
            return redirect('login-page')
    else:
        return render(request, 'todoapp/login.html', {})

def logout_user(request):
    logout(request)
    return redirect('login-page') 

@login_required(redirect_field_name="login-page")
def delete_task(request, task):
    get_todo = todo.objects.get(user=request.user, todo_name=task)
    get_todo.delete()
    return redirect('home-page')

@login_required(redirect_field_name="login-page")
def update_task(request, task):
    get_todo = todo.objects.get(user=request.user, todo_name=task)
    get_todo.status = True 
    get_todo.save()
    return redirect('home-page')



