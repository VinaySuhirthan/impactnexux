#!/usr/bin/env python3
"""
Test script for Qwen 3.5:2B model via Ollama API
Sends a large prompt to test the model's response capabilities
"""
import requests
import json
import time

def test_qwen_large_prompt():
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"

    # Large test prompt
    prompt = """
Hello, can you tell me what 2+2 equals?
"""

    # Request payload
    payload = {
        "model": "qwen3.5:9b",
        "prompt": prompt,
        "stream": False,  # Set to True for streaming responses
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 2048  # Allow longer responses
        }
    }

    print("🚀 Testing Qwen 3.5:2B with large prompt...")
    print(f"📝 Prompt length: {len(prompt)} characters")
    print(f"🔗 API URL: {url}")
    print(f"🤖 Model: {payload['model']}")
    print("\n" + "="*80)

    try:
        start_time = time.time()

        # Send request
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout

        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')

            print("✅ SUCCESS!")
            print(f"⏱️ Response time: {response_time:.2f} seconds")
            print(f"📊 Response length: {len(generated_text)} characters")
            print("\n" + "="*80)
            print("🤖 Qwen Response:")
            print("="*80)
            print(generated_text)
            print("\n" + "="*80)

        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
        print("\n💡 Make sure Ollama is running with:")
        print("   ollama serve")
        print("   ollama pull qwen3.5:2b")

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    test_qwen_large_prompt()