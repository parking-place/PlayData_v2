from django import forms 
from django.contrib.auth.forms import UserCreationForm

# Reordering Form and View
class PositionForm(forms.Form):
    position = forms.CharField() 


