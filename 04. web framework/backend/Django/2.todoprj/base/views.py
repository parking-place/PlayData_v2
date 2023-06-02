from typing import Any, Dict
from django.forms.models import BaseModelForm
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render, redirect 
from django.views.generic.list import ListView 
from django.views.generic.detail import DetailView 
from django.views.generic.edit import CreateView, UpdateView, DeleteView, FormView 
from django.urls import reverse_lazy

from django.contrib.auth.views import LoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login

# Imports for Reordering Feature
from django.views import View
from django.shortcuts import redirect
from django.db import models, transaction

from .models import Task
from .forms import PositionForm

# Create your views here.
class CustomLoginView(LoginView):
    template_name = 'base/login.html'
    fields = '__all__'
    redirect_authenticated_user = True 

    def get_success_url(self) -> str:
        return reverse_lazy('tasks')
    
class RegisterPage(FormView):
    template_name = 'base/register.html'
    form_class = UserCreationForm
    redirect_authenticated_user = True 
    success_url = reverse_lazy('tasks')

    def form_valid(self, form: Any) -> HttpResponse:
        user = form.save() 
        if user is not None:
            login(self.request, user)
        
        return super().form_valid(form)
    
    def get(self, request: HttpRequest, *args: str, **kwargs: Any) -> HttpResponse:
        if self.request.user.is_authenticated:
            return redirect('tasks')
        return super().get(request, *args, **kwargs)
    

class TaskList(LoginRequiredMixin, ListView):
    model = Task 
    context_object_name = 'tasks'
    login_url = 'login/'

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context['tasks'] = context['tasks'].filter(user=self.request.user)
        context['count'] = context['tasks'].filter(complete=False).count()

        search_input = self.request.GET.get('search-area') or ''
        if search_input:
            context['tasks'] = context['tasks'].filter(title__contains=search_input)

        context['search_input'] = search_input 

        return context 
    
class TaskDetail(LoginRequiredMixin, DetailView):
    model = Task 
    context_object_name = 'task'
    template_name = 'base/task.html'
    login_url = 'login/'

class TaskCreate(LoginRequiredMixin, CreateView):
    model = Task 
    fields = ['title', 'description', 'complete']
    success_url = reverse_lazy('tasks')
    login_url = 'login/'

    def form_valid(self, form: BaseModelForm) -> HttpResponse:
        form.instance.user = self.request.user 
        return super().form_valid(form)
    
class TaskUpdate(LoginRequiredMixin, UpdateView):
    model = Task 
    fields = ['title', 'description', 'complete']
    success_url = reverse_lazy('tasks')
    login_url = 'login/'

class DeleteView(LoginRequiredMixin, DeleteView):
    model = Task 
    context_object_name = 'task'
    success_url = reverse_lazy('tasks')
    login_url = 'login/'

    def get_queryset(self) :
        owner = self.request.user 
        return self.model.objects.filter(user=owner)
    
class TaskReorder(View):
    def post(self, request):
        form = PositionForm(request.POST)

        if form.is_valid():
            positionList = form.cleaned_data['position'].split(',')

            with transaction.atomic():
                self.request.user.set_task_order(positionList)
        
        return redirect(reverse_lazy('tasks'))
    


