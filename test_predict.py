# test_predict.py
import requests
BASE = "http://127.0.0.1:5001"

samples = [
    "I absolutely love this!",
    "It is awful and disappointing.",
    "Not sure, it's okay I guess.",
    "Amazing product, would buy again."
]

for s in samples:
    r = requests.post(f"{BASE}/predict", json={"text": s})
    print("Input:", s)
    print("Status:", r.status_code, "Response:", r.json())
    print("-" * 40)
