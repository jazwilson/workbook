import requests

url = 'http://localhost:9696/predict'
customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10} #gunicorn
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10} #docker
print(requests.post(url, json=customer).json())

