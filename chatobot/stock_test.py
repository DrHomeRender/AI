import requests
import json

# NH OpenAPI 인증 정보 (발급받은 AppKey와 AppSecret 입력)
APP_KEY = "YOUR_APP_KEY"
APP_SECRET = "YOUR_APP_SECRET"

# 1. 토큰 발급 (OAuth 2.0)
def get_access_token():
    url = "https://openapi.nhinvestment.com:9443/oauth2/token"
    headers = {"Content-Type": "application/json"}
    data = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"토큰 발급 실패: {response.status_code}")

# 2. 주식 시세 조회 (코스닥 상승 상위 3개)
def get_top_gainers(access_token):
    url = "https://openapi.nhinvestment.com:9443/stock/v1/market/top"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }

    params = {
        "market": "KOSDAQ",
        "sort": "change_rate",
        "order": "desc",
        "count": 3
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        stocks = response.json()["result"]
        for stock in stocks:
            print(f"{stock['stock_name']} - {stock['price']}원 (변동률: {stock['change_rate']}%)")
    else:
        print(f"주식 시세 조회 실패: {response.status_code}")

# 실행
try:
    token = get_access_token()
    print("📈 코스닥 상승 상위 3개:")
    get_top_gainers(token)
except Exception as e:
    print(f"🚨 오류 발생: {e}")
