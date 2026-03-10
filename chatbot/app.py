import os
import json
import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")

@router.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            return JSONResponse({"error": "No message provided"}, status_code=400)

        prompt = f"User: {user_message}\nAssistant:"
        
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        }
        
        print(f"[CHATBOT:OLLAMA] Sending prompt to {MODEL}")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        
        result = resp.json()
        reply = result.get("response", "").strip()
        
        return JSONResponse({"reply": reply})

    except Exception as e:
        print(f"[CHATBOT:OLLAMA] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
