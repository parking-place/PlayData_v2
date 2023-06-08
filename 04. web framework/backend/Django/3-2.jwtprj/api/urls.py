from django.urls import path 
from rest_framework_simplejwt import views as jwt_views 
from .views import HelloWorld, Extractor

urlpatterns = [
    path('login/', jwt_views.TokenObtainPairView.as_view(), name='login'),
    path('run/', HelloWorld.as_view(), name='hello-world'),
    path('me/', Extractor.as_view(), name='extract-token')
]