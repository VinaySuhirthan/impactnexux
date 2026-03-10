import os
import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

@router.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            return JSONResponse({"error": "No message provided"}, status_code=400)

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful marketing AI assistant for AdGPT."},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        
        print(f"[CHATBOT:GROQ] Sending request to {GROQ_MODEL}")
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        
        return JSONResponse({"reply": reply})

    except Exception as e:
        print(f"[CHATBOT:GROQ] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
