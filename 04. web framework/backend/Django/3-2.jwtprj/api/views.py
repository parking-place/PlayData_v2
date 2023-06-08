from django.shortcuts import render
from rest_framework.views import APIView 
from rest_framework.response import Response 
from rest_framework.permissions import IsAuthenticated

# Create your views here.
# HelloWorld, Extractor
class HelloWorld(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        content = {
            "message": "Hello World"
        }
        return Response(content)
    
class Extractor(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        content = {
            "username": request.user.username,
            "password": request.user.password
        }
        return Response(content)