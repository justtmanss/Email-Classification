# Create a file named test_api.py
import requests
import json

# API URL - change if testing locally or deployed version
API_URL = "http://localhost:8000"  # Change to your Hugging Face URL when deployed

# Test the root endpoint
def test_root():
    response = requests.get(f"{API_URL}/")
    print("Root Endpoint Response:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

# Test email classification
def test_classify_email():
    test_data = {
        "email_body": "Hello support, my name is Sarah Johnson and I've been charged twice for my subscription. My account email is sarah.j@example.com and my phone is 555-987-6543. Please help resolve this billing issue."
    }
    
    response = requests.post(f"{API_URL}/classify-email", json=test_data)
    
    print("Email Classification Response:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

if __name__ == "__main__":
    print("Testing API Endpoints...")
    test_root()
    test_classify_email()