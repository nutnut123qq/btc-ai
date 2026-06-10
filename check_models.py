import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
data = r.json()
if "models" in data:
    for m in data["models"]:
        print(m["name"])
else:
    print(data)
