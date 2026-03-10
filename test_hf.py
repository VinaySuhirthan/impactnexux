import requests
import io
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def test_hf():
    prompt = "a professional ad for a fitness gym, cinematic"
    print(f"Testing HF API with prompt: {prompt}")
    payload = {"inputs": prompt}
    response = requests.post(MODEL_URL, headers=headers, json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Success! Got image.")
        img = Image.open(io.BytesIO(response.content))
        img.save("test_hf.png")
        print("Saved to test_hf.png")
    else:
        print(f"Failed. Status: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    test_hf()
