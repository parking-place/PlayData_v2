from django.test import TestCase
from rest_framework.test import APIClient, APITestCase 
from django.urls import reverse 
from django.contrib.auth.models import User 

# Create your tests here.
class AuthenticatorTestCase(APITestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user("tester", "test@email.com", "tester")
        self.client = APIClient()
        self.login_url = reverse('login')
        self.hellworld_url = reverse('hello-world')
        self.extractor_url = reverse('extract-token')
        self.access_token = self.client.post(self.login_url, {"username": "tester",
            "password": "tester"}).json()['access']
    
    def test_login_return_jwt(self):
        """
        The login view return an acceess token and a refresh token
        """
        credentials = {
            "username": "tester",
            "password": "tester"
        }

        response = self.client.post(self.login_url, credentials)
        print("------ test_login_return_jwt ------")
        print(response.status_code)
        print(response.json())
        self.assertEqual(response.status_code, 200)
        self.assertTrue('access' in response.json().keys())
        self.assertTrue('refresh' in response.json().keys())

    def test_bad_login_dont_return_jwt(self):
        """
        The login view doesn't return tokens if we use bad credentials
        """
        credentials = {
            "username": "bad tester",
            "password": "bad tester"
        }

        response = self.client.post(self.login_url, credentials)
        print("------ test_bad_login_dont_return_jwt ------")
        print(response.status_code)
        print(response.json())
        self.assertEqual(response.status_code, 401)

    def test_run_return_hellowrld(self):
        """
        The helloworld view return hello world 
        """
        self.client.credentials(HTTP_AUTHORIZATION='Bearer '+self.access_token)
        response = self.client.get(self.hellworld_url)
        print("------ test_run_return_hellowrld ------")
        print(response.status_code)
        print(response.json())

        expected_response = {'message':'Hello World'}
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected_response)

