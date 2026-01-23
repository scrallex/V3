import requests
import json

print("--- /api/account ---")
try:
    r = requests.get("http://sep-backend:8000/api/account", timeout=3)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print(e)

print("\n--- /api/state (REDIS via Frontend) ---")
# Frontend is on port 80 locally inside sep-frontend, but we are running this likely in backend or temp.
# We can hit sep-frontend:80 or internal localhost:8000 if in frontend.
# Let's try hitting sep-frontend container from outside or backend.
try:
    r = requests.get("http://sep-frontend:80/api/state", timeout=3)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print(e)
