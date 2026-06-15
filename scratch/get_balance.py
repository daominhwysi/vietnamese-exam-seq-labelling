import os
import requests

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable is not set.")
        return

    url = "https://api.deepseek.com/user/balance"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    try:
        response = requests.get(url, headers=headers)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
    except Exception as e:
        print("An error occurred during the request:", e)

if __name__ == "__main__":
    main()
