import jwt, datetime 
from rest_framework.exceptions import AuthenticationFailed

def create_access_token(id):
    return jwt.encode({
        'user_id': id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=60), # expired date
        'iat': datetime.datetime.utcnow() # created date 
    }, 'access_secret', algorithm='HS256')

def decode_access_token(token):
    try:
        payload = jwt.decode(token, 'access_secret', algorithms='HS256')

        return payload['user_id']
    except Exception as e:
        print('aaaaa')
        print(e)
        raise AuthenticationFailed('unauthenticated')

def create_refresh_token(id):
    return jwt.encode({
        'user_id': id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7), # expired date
        'iat': datetime.datetime.utcnow() # created date 
    }, 'refresh_secret', algorithm='HS256')

