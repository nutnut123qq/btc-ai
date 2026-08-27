import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in .env or environment.")

r = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    headers={"x-goog-api-key": key},
    timeout=10,
)
data = r.json()
if "models" in data:
    for m in data["models"]:
        print(m["name"])
else:
    print(data)
