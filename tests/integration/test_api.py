import requests
from pathlib import Path

# Test API endpoint
url = "http://13.229.232.95:3000/scan_faces"

# Test image path
image_path = Path("images/IMG_4793 Medium.jpeg")

# Send request
files = [('file', (image_path.name, open(str(image_path), 'rb'), 'application/octet-stream'))]
headers = {
    'Accept': 'application/json',
    'User-Agent': 'FaceRecognition/1.0'
}

try:
    response = requests.post(url, headers=headers, files=files, timeout=30)
    response.raise_for_status()
    print("API Response:", response.json())
except Exception as e:
    print("Error:", str(e))
