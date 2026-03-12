from pathlib import Path
import json
import os
import sys
import re as _re
import time
from typing import Any, Dict, List, Optional
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess

# Import chatbot router
sys.path.append(str(Path(__file__).resolve().parent.parent))
from chatbot.app import router as chatbot_router

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
INDEX_FILE = BASE_DIR / "index.html"
IMAGE_FILE = BASE_DIR / "image.html"
MAIN_FILE = ROOT_DIR / "main.html"

# Serve static files (generated images)
if not os.path.exists(ROOT_DIR / "static" / "generated"):
    os.makedirs(ROOT_DIR / "static" / "generated", exist_ok=True)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")

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

CATEGORY_FLOW.append(
    {
        "id": "contact",
        "label": "Contact Details",
        "fixed_question": "How can customers reach you? (At least one is required)",
        "contact_type": True,
    }
)

FIELD_MAP = {f["id"]: f for f in CATEGORY_FLOW}

@app.get("/")
async def get_main():
    return FileResponse(MAIN_FILE)

@app.get("/generator")
async def get_generator():
    return FileResponse(INDEX_FILE)

@app.get("/image")
async def get_image_page():
    return FileResponse(IMAGE_FILE)

@app.get("/logo.jpeg")
async def get_logo():
    return FileResponse(ROOT_DIR / "logo.jpeg")

@app.get("/chatbot")
async def get_chatbot_page():
    return FileResponse(ROOT_DIR / "chatbot" / "chat.html")

app.include_router(chatbot_router)

# ── Utility helpers ──────────────────────────────────────────────────────────

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


def _fallback_extract_qa(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    question = ""; options = []
    for line in lines:
        if line.endswith("?") and not question:
            question = line.lstrip("- *#").strip()
        elif _re.match(r'^[\d]+[.)\-]\s+', line):
            opt = _re.sub(r'^[\d]+[.)\-]\s+', '', line).strip().strip('"\'')
            if 2 <= len(opt) <= 80: options.append(opt)
        elif line.startswith(("-", "•", "*")) and not line.endswith("?"):
            opt = line.lstrip("-•* ").strip().strip('"\'')
            if 2 <= len(opt) <= 80: options.append(opt)
    if question and len(options) >= 2:
        return {"question": question, "options": options[:5]}
    return None

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

def _check_ollama_health() -> bool:
    global OLLAMA_LAST_HEALTH, OLLAMA_HEALTH_CHECK_TIME, OLLAMA_AVAILABLE_MODELS
    if not OLLAMA_ENABLED: return False
    now = time.time()
    if now - OLLAMA_HEALTH_CHECK_TIME < 30: return OLLAMA_LAST_HEALTH
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if resp.status_code == 200:
            OLLAMA_LAST_HEALTH = True
            OLLAMA_HEALTH_CHECK_TIME = now
            try:
                data = resp.json()
                OLLAMA_AVAILABLE_MODELS = sorted([m["name"] for m in data.get("models", [])])
            except: pass
            return True
    except: pass
    OLLAMA_LAST_HEALTH = False
    OLLAMA_HEALTH_CHECK_TIME = now
    return False

def ask_groq(prompt: str) -> Dict[str, Any]:
    global ACTIVE_MODEL
    now = time.time()
    if not GROQ_ENABLED: return {"text": "", "model": ACTIVE_MODEL, "error": "Groq not hexed."}
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": GROQ_MODEL, "messages": [{"role":"user","content":prompt}], "temperature":0.7, "max_tokens":1024}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        ACTIVE_MODEL = GROQ_MODEL
        return {"text": text, "model": GROQ_MODEL, "error": ""}
    except Exception as exc:
        return {"text": "", "model": ACTIVE_MODEL, "error": str(exc)}

def ask_llm(prompt: str) -> Dict[str, Any]:
    global ACTIVE_MODEL
    if not OLLAMA_ENABLED:
        if GROQ_FALLBACK_ENABLED and GROQ_ENABLED: return ask_groq(prompt)
        return {"text": "", "model": ACTIVE_MODEL, "error": "Ollama disabled."}
    try:
        payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": 1024}}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        _clear_ollama_error()
        ACTIVE_MODEL = MODEL
        return {"text": text, "model": MODEL, "error": ""}
    except Exception as e:
        if GROQ_FALLBACK_ENABLED and GROQ_ENABLED: return ask_groq(prompt)
        return {"text": "", "model": ACTIVE_MODEL, "error": str(e)}

@app.get("/api/status")
def api_status():
    ollama_health = _check_ollama_health()
    return JSONResponse({
        "status": "ok",
        "groq": {"enabled": GROQ_ENABLED, "model": GROQ_MODEL},
        "ollama": {
            "enabled": OLLAMA_ENABLED,
            "configured_model": MODEL,
            "active_model": ACTIVE_MODEL,
            "healthy": ollama_health,
            "url": OLLAMA_BASE_URL,
            "available_models": OLLAMA_AVAILABLE_MODELS,
            "last_error": OLLAMA_LAST_ERROR,
            "retry_after_seconds": max(0, int(round(OLLAMA_DISABLED_UNTIL - time.time()))) if OLLAMA_DISABLED_UNTIL>0 else 0,
        }
    })

@app.post("/api/next_question")
async def api_next_question(request: Request):
    data = await request.json()
    answers = data.get("answers", {})
    field_id = next_field_id(answers)
    if field_id is None: return JSONResponse({"done": True})
    field = FIELD_MAP[field_id]
    
    if "fixed_question" in field:
        return JSONResponse({
            "done": False, "field": field_id, "label": field["label"],
            "question": field["fixed_question"], "options": field.get("fixed_options", []),
            "contact_type": field.get("contact_type", False),
            "ollama": _ollama_status(),
        })
    
    # Simple dynamic prompt for brevity
    prompt = f"Senior strategist brief. Answers so far: {json.dumps(answers)}. Ask one punchy 10-word strategic question. 5 options. JSON: {{\"question\":\"...\",\"options\":[\"...\"]}}"
    result = ask_llm(prompt)
    if result["error"]: return JSONResponse({"error": result["error"], "ollama": _ollama_status()})
    obj = _parse_json_object(result["text"]) or _fallback_extract_qa(result["text"])
    if not obj: return JSONResponse({"error": "LLM Parse Failure", "ollama": _ollama_status()})
    
    return JSONResponse({
        "done": False, "field": field_id, "label": field["label"],
        "question": sanitize_question(obj.get("question", "")),
        "options": _clean_options(obj.get("options", [])),
        "ollama": _ollama_status(),
    })

@app.post("/api/generate_assets")
async def api_generate_assets(request: Request):
    data = await request.json()
    answers = data.get("answers", {})
    prompt = f"Expert copywriter. Brief: {json.dumps(answers)}. Create 5 Headlines, 5 Taglines, 5 CTAs, 5 Image Prompts."
    result = ask_llm(prompt)
    if result["error"]: return JSONResponse({"error": result["error"], "ollama": _ollama_status()})
    return JSONResponse({"output": result["text"], "ollama": _ollama_status()})

@app.post("/api/generate_image")
async def api_generate_image(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    try:
        result = subprocess.run([sys.executable, str(BASE_DIR / "imagegen.py"), prompt], capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split("\n"):
            if line.strip().startswith('{"images":'): return JSONResponse(json.loads(line.strip()))
        return JSONResponse({"error": "Bad Output"}, status_code=500)
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/generate_video")
async def api_generate_video(request: Request):
    payload = await request.json()
    prompt = payload.get("prompt", "")
    image_path = payload.get("image_path", "")

    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    try:
        # Construct command (now in same directory)
        cmd = [sys.executable, str(BASE_DIR / "videogen.py"), prompt]
        if image_path:
            abs_img_path = str(ROOT_DIR / image_path) if not os.path.isabs(image_path) else image_path
            cmd.extend(["--images", abs_img_path])

        print(f"[VIDEO] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            cwd=str(ROOT_DIR)
        )

        video_output_name = "final_ad_film.mp4"
        output_src = ROOT_DIR / video_output_name
        
        if not output_src.exists():
            return JSONResponse({"error": "Video generation script completed but output file missing"}, status_code=500)

        # Move to static/generated
        timestamp = int(time.time())
        new_filename = f"video_{timestamp}.mp4"
        dest_path = ROOT_DIR / "static" / "generated" / new_filename
        os.rename(output_src, dest_path)

        return JSONResponse({
            "video_url": f"/static/generated/{new_filename}",
            "message": "Video generated successfully"
        })

    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"Video generation failed: {e.stderr}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)