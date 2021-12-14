import requests

host = 'localhost:9696'
url = f'http://{host}/predict'

school = {
 'type_Public': 1,
 'state_Texas':1
}

response = requests.post(url, json=school).json()
print(response)