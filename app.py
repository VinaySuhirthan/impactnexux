from pathlib import Path
import json
import os
import re as _re
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BASE_URL = (
    OLLAMA_URL.split("/api/", 1)[0].rstrip("/")
    if "/api/" in OLLAMA_URL
    else OLLAMA_URL.rstrip("/")
)
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_ENABLED = (
    os.getenv("OLLAMA_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
)
OLLAMA_RETRY_SECONDS = int(os.getenv("OLLAMA_RETRY_SECONDS", "5"))
OLLAMA_DISABLED_UNTIL = 0.0
OLLAMA_LAST_ERROR = ""
OLLAMA_AVAILABLE_MODELS: List[str] = []
OLLAMA_HEALTH_CHECK_TIME = 0.0
OLLAMA_LAST_HEALTH = False
ACTIVE_MODEL = MODEL

# ── Groq configuration (used as fallback when Ollama fails) ───────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_ENABLED = bool(GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"))
GROQ_RETRY_SECONDS = int(os.getenv("GROQ_RETRY_SECONDS", "5"))
GROQ_DISABLED_UNTIL = 0.0
GROQ_LAST_ERROR = ""
GROQ_FALLBACK_ENABLED = os.getenv("GROQ_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

TOTAL_DYNAMIC_STEPS = 4

CATEGORY_FLOW = [
    {
        "id": "product",
        "label": "Product / Service",
        "fixed_question": "What product or service are you advertising?",
        "fixed_options": [
            "Protein or nutrition product",
            "SaaS software",
            "Coaching or course",
            "Local business service",
            "Mobile app",
        ],
    },
]
for i in range(1, TOTAL_DYNAMIC_STEPS + 1):
    CATEGORY_FLOW.append(
        {
            "id": f"step_{i}",
            "label": f"Dynamic Step {i}",
        }
    )

# Add contact details field
CATEGORY_FLOW.append(
    {
        "id": "contact",
        "label": "Contact Details",
        "fixed_question": "How can customers reach you? (At least one is required)",
        "contact_type": True,
    }
)

FIELD_MAP = {f["id"]: f for f in CATEGORY_FLOW}


# ── Utility helpers (identical to app.py) ─────────────────────────────────────

def has_value(data: dict, key: str) -> bool:
    value = data.get(key, "")
    return isinstance(value, str) and value.strip() != ""


def next_field_id(answers: dict) -> Optional[str]:
    for field in CATEGORY_FLOW:
        if not has_value(answers, field["id"]):
            return field["id"]
    return None


def sanitize_question(text: str) -> str:
    if not text:
        return ""
    line = text.strip().replace("\n", " ").replace("\r", " ")
    line = " ".join(line.split())
    line = line.strip(' "\'')
    if len(line) < 8:
        return ""
    if len(line) > 140:
        line = line[:140].rstrip()
    if not line.endswith("?"):
        line = line.rstrip(".!") + "?"
    return line


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _clean_options(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        option = " ".join(value.strip().split())
        if len(option) < 2:
            continue
        key = option.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(option)
        if len(cleaned) == 5:
            break
    return cleaned


# ── Ollama error state helpers ────────────────────────────────────────────────

def _set_ollama_error(message: str, cooldown_seconds: Optional[int] = None) -> None:
    global OLLAMA_DISABLED_UNTIL, OLLAMA_LAST_ERROR
    OLLAMA_LAST_ERROR = message
    if cooldown_seconds is not None:
        OLLAMA_DISABLED_UNTIL = time.time() + max(1, cooldown_seconds)


def _clear_ollama_error() -> None:
    global OLLAMA_DISABLED_UNTIL, OLLAMA_LAST_ERROR
    OLLAMA_DISABLED_UNTIL = 0.0
    OLLAMA_LAST_ERROR = ""


def _ollama_status() -> Dict[str, Any]:
    retry_after = 0
    if OLLAMA_DISABLED_UNTIL > 0:
        retry_after = max(0, int(round(OLLAMA_DISABLED_UNTIL - time.time())))

    return {
        "enabled": OLLAMA_ENABLED,
        "configured_model": MODEL,
        "active_model": ACTIVE_MODEL,
        "last_error": OLLAMA_LAST_ERROR,
        "retry_after_seconds": retry_after,
    }


# ── Groq error state helpers ──────────────────────────────────────────────────

def _set_groq_error(message: str, cooldown_seconds: Optional[int] = None) -> None:
    global GROQ_DISABLED_UNTIL, GROQ_LAST_ERROR
    GROQ_LAST_ERROR = message
    if cooldown_seconds is not None:
        GROQ_DISABLED_UNTIL = time.time() + max(1, cooldown_seconds)


def _clear_groq_error() -> None:
    global GROQ_DISABLED_UNTIL, GROQ_LAST_ERROR
    GROQ_DISABLED_UNTIL = 0.0
    GROQ_LAST_ERROR = ""


# ── Ollama health check ───────────────────────────────────────────────────────

def _check_ollama_health() -> bool:
    """Check if Ollama is running and accessible (cached for 30s)."""
    global OLLAMA_LAST_HEALTH, OLLAMA_HEALTH_CHECK_TIME, OLLAMA_AVAILABLE_MODELS
    
    if not OLLAMA_ENABLED:
        return False
    
    now = time.time()
    # Use cache if recent (30 seconds)
    if now - OLLAMA_HEALTH_CHECK_TIME < 30:
        return OLLAMA_LAST_HEALTH
    
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if resp.status_code == 200:
            OLLAMA_LAST_HEALTH = True
            OLLAMA_HEALTH_CHECK_TIME = now
            # Also cache models while we're at it
            try:
                data = resp.json()
                OLLAMA_AVAILABLE_MODELS = sorted([m["name"] for m in data.get("models", [])])
            except Exception:
                pass
            return True
    except Exception:
        pass
    
    OLLAMA_LAST_HEALTH = False
    OLLAMA_HEALTH_CHECK_TIME = now
    return False


def _get_available_models() -> List[str]:
    """Fetch list of available models from Ollama."""
    if not OLLAMA_ENABLED:
        return []
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return sorted(models)
    except Exception as e:
        print(f"[BRAIN:OLLAMA] Failed to fetch models: {e}")
        return []


# ── Core LLM call ─────────────────────────────────────────────────────────────

def ask_groq(prompt: str) -> Dict[str, Any]:
    """Send prompt to Groq API and return similar dict structure."""
    global ACTIVE_MODEL
    now = time.time()
    if now < GROQ_DISABLED_UNTIL:
        wait_seconds = max(1, int(round(GROQ_DISABLED_UNTIL - now)))
        message = GROQ_LAST_ERROR or f"Groq is cooling down. Retry in {wait_seconds}s."
        return {"text": "", "model": ACTIVE_MODEL, "error": message}
    if not GROQ_ENABLED:
        return {"text": "", "model": ACTIVE_MODEL, "error": "Groq not configured."}
    try:
        print(f"[BRAIN:GROQ] Sending REST prompt to '{GROQ_MODEL}' ({len(prompt)} chars)")
        print(f"[BRAIN:GROQ] Prompt preview: {prompt[:120].strip()!r}...")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role":"user","content":prompt}],
            "temperature":0.7,
            "max_tokens":1024,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ValueError(f"Invalid Groq response format: {e}")
        if not text:
            raise ValueError("Groq returned empty text")
        ACTIVE_MODEL = GROQ_MODEL
        _clear_groq_error()
        print(f"[BRAIN:GROQ] Response received ({len(text)} chars): {text[:120]!r}...")
        return {"text": text, "model": GROQ_MODEL, "error": ""}
    except requests.HTTPError as exc:
        message = f"Groq HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        print(f"[BRAIN:GROQ] HTTP Error: {message}")
        _set_groq_error(message, cooldown_seconds=GROQ_RETRY_SECONDS)
        return {"text": "", "model": ACTIVE_MODEL, "error": message}
    except Exception as exc:
        message = f"Groq error: {exc}"
        print(f"[BRAIN:GROQ] Exception: {exc}")
        _set_groq_error(message, cooldown_seconds=GROQ_RETRY_SECONDS)
        return {"text": "", "model": ACTIVE_MODEL, "error": message}


def ask_llm(prompt: str) -> Dict[str, Any]:
    global ACTIVE_MODEL

    if not OLLAMA_ENABLED:
        if GROQ_FALLBACK_ENABLED and GROQ_ENABLED:
            print("[BRAIN:LLM] Ollama disabled, using Groq")
            return ask_groq(prompt)
        return {
            "text": "",
            "model": ACTIVE_MODEL,
            "error": "Ollama is disabled and Groq fallback is disabled.",
        }

    now = time.time()
    if now < OLLAMA_DISABLED_UNTIL:
        wait_seconds = max(1, int(round(OLLAMA_DISABLED_UNTIL - now)))
        message = OLLAMA_LAST_ERROR or f"Ollama is cooling down. Retry in {wait_seconds}s."
        if GROQ_FALLBACK_ENABLED and GROQ_ENABLED:
            print(f"[BRAIN:LLM] {message}, falling back to Groq")
            return ask_groq(prompt)
        return {"text": "", "model": ACTIVE_MODEL, "error": message}

    try:
        print(f"[BRAIN:OLLAMA] Sending REST prompt to '{MODEL}' ({len(prompt)} chars)")
        print(f"[BRAIN:OLLAMA] Prompt preview: {prompt[:120].strip()!r}...")
        
        payload = {"model": MODEL, "prompt": prompt, "stream": False}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        resp.raise_for_status()
        
        data = resp.json()
        text = data.get("response", "").strip()
        print(f"[BRAIN:OLLAMA] Response received ({len(text)} chars): {text[:120]!r}...")
        
        if not text:
            _set_ollama_error("Empty response from Ollama", OLLAMA_RETRY_SECONDS)
            if GROQ_FALLBACK_ENABLED and GROQ_ENABLED:
                print("[BRAIN:OLLAMA] Empty response, falling back to Groq")
                return ask_groq(prompt)
            return {"text": "", "model": ACTIVE_MODEL, "error": "Empty response from Ollama"}
        
        _clear_ollama_error()
        ACTIVE_MODEL = MODEL
        return {"text": text, "model": MODEL, "error": ""}
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Ollama request failed: {str(e)}"
        print(f"[BRAIN:OLLAMA] {error_msg}")
        _set_ollama_error(error_msg, OLLAMA_RETRY_SECONDS)
        if GROQ_FALLBACK_ENABLED and GROQ_ENABLED:
            print("[BRAIN:OLLAMA] falling back to Groq")
            return ask_groq(prompt)
        return {"text": "", "model": ACTIVE_MODEL, "error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error with Ollama: {str(e)}"
        print(f"[BRAIN:OLLAMA] {error_msg}")
        _set_ollama_error(error_msg, OLLAMA_RETRY_SECONDS)
        if GROQ_FALLBACK_ENABLED and GROQ_ENABLED:
            print("[BRAIN:OLLAMA] unexpected error, falling back to Groq")
            return ask_groq(prompt)
        return {"text": "", "model": ACTIVE_MODEL, "error": error_msg}

# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(INDEX_FILE)


@app.get("/api/status")
def api_status():
    ollama_health = _check_ollama_health()
    
    return JSONResponse({
        "status": "ok",
        "version": "1.0",
        "groq": {
            "enabled": GROQ_ENABLED,
            "model": GROQ_MODEL,
        },
        "ollama": {
            "enabled": OLLAMA_ENABLED,
            "configured_model": MODEL,
            "active_model": ACTIVE_MODEL,
            "healthy": ollama_health,
            "url": OLLAMA_BASE_URL,
            "available_models": OLLAMA_AVAILABLE_MODELS,
            "last_error": OLLAMA_LAST_ERROR,
            "retry_after_seconds": max(0, int(round(OLLAMA_DISABLED_UNTIL - time.time()))) if OLLAMA_DISABLED_UNTIL > 0 else 0,
        }
    })


@app.post("/api/next_question")
async def api_next_question(request: Request):
    data = await request.json()
    answers = data.get("answers", {})
    
    field_id = next_field_id(answers)
    if field_id is None:
        return JSONResponse({"done": True})
    
    field = FIELD_MAP[field_id]
    
    if field_id == "contact":
        # Contact field special handling
        return JSONResponse({
            "done": False,
            "field": field_id,
            "label": field["label"],
            "question": field["fixed_question"],
            "contact_type": True,
            "total_steps": len(CATEGORY_FLOW),
            "answered_steps": len([k for k in answers if has_value(answers, k)]),
            "ollama": _ollama_status(),
        })
    
    if "fixed_question" in field:
        # Fixed question
        return JSONResponse({
            "done": False,
            "field": field_id,
            "label": field["label"],
            "question": field["fixed_question"],
            "options": field["fixed_options"],
            "total_steps": len(CATEGORY_FLOW),
            "answered_steps": len([k for k in answers if has_value(answers, k)]),
            "ollama": _ollama_status(),
        })
    
    # Dynamic question
    prompt = f"""
You are an expert marketing strategist helping create targeted advertising campaigns.

Based on the following user answers, generate the next most important question to ask for creating effective ad copy.

User answers so far:
{json.dumps(answers, indent=2)}

Generate a single question that will help refine the ad targeting and messaging. The question should be specific and actionable, focusing on key aspects like target audience, unique value proposition, pain points, or competitive advantages.

Return your response as a JSON object with:
- "question": the question text (max 140 chars, ending with ?)
- "options": array of 5 specific answer options (each 2-50 chars)

Example:
{{
  "question": "What is your target audience's primary pain point?",
  "options": ["High costs", "Time waste", "Poor quality", "Lack of access", "Complex processes"]
}}

Make the question and options highly relevant to advertising and marketing, aiming to build a complete understanding efficiently.
""".strip()
    
    result = ask_llm(prompt)
    if result["error"]:
        return JSONResponse({"error": result["error"], "ollama": _ollama_status()})
    
    text = result["text"]
    obj = _parse_json_object(text)
    if not obj:
        return JSONResponse({"error": "Failed to parse LLM response", "ollama": _ollama_status()})
    
    question = sanitize_question(obj.get("question", ""))
    options = _clean_options(obj.get("options", []))
    
    if not question or len(options) < 2:
        return JSONResponse({"error": "Invalid question/options from LLM", "ollama": _ollama_status()})
    
    return JSONResponse({
        "done": False,
        "field": field_id,
        "label": field["label"],
        "question": question,
        "options": options,
        "total_steps": len(CATEGORY_FLOW),
        "answered_steps": len([k for k in answers if has_value(answers, k)]),
        "ollama": _ollama_status(),
    })


@app.post("/api/generate_assets")
async def api_generate_assets(request: Request):
    data = await request.json()
    answers = data.get("answers", {})
    
    if not answers:
        return JSONResponse({"error": "No answers provided"})
    
    prompt = f"""
You are a professional copywriter creating advertising assets.

Based on the following user answers, generate compelling ad copy including headlines, taglines, and CTAs.

User answers:
{json.dumps(answers, indent=2)}

Generate:
- 5 Headlines (each 5-30 words)
- 5 Taglines (each 3-10 words)  
- 5 Call-to-Action phrases (each 2-8 words)
- 5 Image prompt descriptions (each 10-50 words for AI image generation)

Format your response as:
HEADLINES:
1. Headline one
2. Headline two
...

TAGLINES:
1. Tagline one
2. Tagline two
...

CTAS:
1. CTA one
2. CTA two
...

IMAGE_PROMPTS:
1. Image prompt one
2. Image prompt two
...

Make everything highly targeted and persuasive for advertising.
""".strip()
    
    result = ask_llm(prompt)
    if result["error"]:
        return JSONResponse({"error": result["error"], "ollama": _ollama_status()})
    
    text = result["text"]
    return JSONResponse({
        "output": text,
        "ollama": _ollama_status(),
    })