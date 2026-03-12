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

# Import chatbot router (assuming chatbot is a package in root)
# We need to add parent to sys.path if running from within question_mode
sys.path.append(str(Path(__file__).resolve().parent.parent))
from chatbot.appbc import router as chatbot_router

# ── Groq configuration ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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
# Note: Since we are in a subfolder, we should use the ROOT static folder.
if not os.path.exists(ROOT_DIR / "static" / "generated"):
    os.makedirs(ROOT_DIR / "static" / "generated", exist_ok=True)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


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

# Include chatbot router
app.include_router(chatbot_router)

# ── State that mirrors app.py's "ollama" state (renamed to "groq") ──────────
GROQ_ENABLED = (
    os.getenv("GROQ_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
)
GROQ_DISABLED_UNTIL = 0.0
GROQ_LAST_ERROR = ""
ACTIVE_MODEL = GROQ_MODEL
GROQ_RETRY_SECONDS = int(os.getenv("GROQ_RETRY_SECONDS", "5"))

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


# ── Groq error state helpers ────────────────────────────────────────────────

def _set_groq_error(message: str, cooldown_seconds: Optional[int] = None) -> None:
    global GROQ_DISABLED_UNTIL, GROQ_LAST_ERROR
    GROQ_LAST_ERROR = message
    if cooldown_seconds is not None:
        GROQ_DISABLED_UNTIL = time.time() + max(1, cooldown_seconds)


def _clear_groq_error() -> None:
    global GROQ_DISABLED_UNTIL, GROQ_LAST_ERROR
    GROQ_DISABLED_UNTIL = 0.0
    GROQ_LAST_ERROR = ""


def _groq_status() -> Dict[str, Any]:
    retry_after = 0
    if GROQ_DISABLED_UNTIL > 0:
        retry_after = max(0, int(round(GROQ_DISABLED_UNTIL - time.time())))

    return {
        "enabled": GROQ_ENABLED,
        "configured_model": GROQ_MODEL,
        "active_model": ACTIVE_MODEL,
        "last_error": GROQ_LAST_ERROR,
        "retry_after_seconds": retry_after,
    }


# ── Core LLM call ─────────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> Dict[str, Any]:
    global ACTIVE_MODEL

    if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
        return {
            "text": "",
            "model": ACTIVE_MODEL,
            "error": "GROQ_API_KEY is not set or invalid. Please set GROQ_API_KEY environment variable with a valid key.",
        }

    if not GROQ_ENABLED:
        return {
            "text": "",
            "model": ACTIVE_MODEL,
            "error": "Groq is disabled. Set GROQ_ENABLED=true to use dynamic questions.",
        }

    now = time.time()
    if now < GROQ_DISABLED_UNTIL:
        wait_seconds = max(1, int(round(GROQ_DISABLED_UNTIL - now)))
        message = GROQ_LAST_ERROR or f"Groq is cooling down. Retry in {wait_seconds}s."
        return {"text": "", "model": ACTIVE_MODEL, "error": message}

    try:
        print(f"[BRAIN:GROQ] Sending REST prompt to '{GROQ_MODEL}' ({len(prompt)} chars)")
        print(f"[BRAIN:GROQ] Prompt preview: {prompt[:120].strip()!r}...")
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        
        try:
            text = data["choices"][0]["message"]["content"]
            text = text.strip() if text else ""
        except (KeyError, IndexError) as e:
            print(f"[BRAIN:GROQ] Parse error: {e}, data: {data}")
            raise ValueError(f"Invalid response format from Groq output: {e}")
            
        if not text:
            raise ValueError("Groq returned an empty response.")
        ACTIVE_MODEL = GROQ_MODEL
        _clear_groq_error()
        print(f"[BRAIN:GROQ] Response received ({len(text)} chars): {text[:120].strip()!r}...")
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


# ── Dynamic step generation (identical logic to app.py) ───────────────────────

def generate_dynamic_step(answers: dict, field_id: str) -> Dict[str, Any]:
    answered_small = {k: v for k, v in answers.items() if isinstance(v, str) and v.strip()}
    remaining_ids = [
        f["id"]
        for f in CATEGORY_FLOW
        if not has_value(answers, f["id"]) and f["id"] != "product"
    ]
    step_number = next(
        (idx for idx, f in enumerate(CATEGORY_FLOW, start=1) if f["id"] == field_id),
        2,
    )

    prompt = f"""
You are a senior brand strategist and growth marketer with MBA-level expertise from top business schools.
You are conducting a structured ad brief interview to design a high-performing marketing campaign.

Your job is to ask the next strategic question that helps define positioning, audience, messaging, or conversion strategy.

Rules:
1) This is interview step {step_number} of {len(CATEGORY_FLOW)}.
2) Ask one precise, strategic marketing question (max 16 words).
3) Generate exactly 5 concise answer options (max 6 words each).
4) Questions must reflect real marketing strategy thinking (targeting, positioning, value proposition, messaging, funnel, platform, pricing perception, differentiation).
5) Options must be practical and realistic choices a marketer would evaluate.
6) Avoid generic or vague questions.
7) Do not include an "Other" option.
8) Do not repeat previously answered topics.
9) Focus on insights that would directly improve ad performance or campaign clarity.
10) Output valid JSON only with this exact shape:
{{"question":"...","options":["...","...","...","...","..."]}}

Product:
{answers.get('product', '').strip()}

Answered data:
{json.dumps(answered_small, ensure_ascii=True)}

Remaining steps:
{', '.join(remaining_ids)}
"""

    result = ask_llm(prompt)
    if not result["text"]:
        return {"error": result["error"], "model": result["model"]}

    raw_text = result["text"]
    raw_text = _re.sub(r"```(?:json)?\s*", "", raw_text).strip().strip("`")

    obj = _parse_json_object(raw_text)
    question = sanitize_question(obj.get("question", "") if isinstance(obj, dict) else "")
    options = _clean_options(obj.get("options") if isinstance(obj, dict) else None)

    if not question or len(options) != 5:
        repair_prompt = f"""
You must repair this output into valid JSON.

Required shape:
{{"question":"...","options":["...","...","...","...","..."]}}

Rules:
- Keep the meaning aligned with the original output.
- Question must end with a question mark.
- Provide exactly 5 options.
- Output JSON only, no markdown fences.

Original output:
{raw_text}
"""
        repaired = ask_llm(repair_prompt)
        if repaired["text"]:
            repaired_text = _re.sub(r"```(?:json)?\s*", "", repaired["text"]).strip().strip("`")
            repaired_obj = _parse_json_object(repaired_text)
            question = sanitize_question(
                repaired_obj.get("question", "") if isinstance(repaired_obj, dict) else ""
            )
            options = _clean_options(
                repaired_obj.get("options") if isinstance(repaired_obj, dict) else None
            )
            if question and len(options) == 5:
                return {"question": question, "options": options, "model": repaired["model"]}

        return {
            "error": "Gemini returned an invalid question payload. Retry the step.",
            "model": result["model"],
        }

    return {"question": question, "options": options, "model": result["model"]}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/ollama_status")
def api_ollama_status():
    return JSONResponse(_groq_status())


@app.post("/api/next_question")
async def api_next_question(request: Request):
    payload = await request.json()
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        answers = {}

    field_id = next_field_id(answers)
    if not field_id:
        return JSONResponse({"done": True, "ollama": _groq_status()})

    field = FIELD_MAP[field_id]
    if field_id == "product":
        question = field["fixed_question"]
        options = field["fixed_options"]
        response_payload = {
            "done": False,
            "field": field_id,
            "label": field["label"],
            "question": question,
            "options": options,
            "total_steps": len(CATEGORY_FLOW),
            "answered_steps": sum(1 for f in CATEGORY_FLOW if has_value(answers, f["id"])),
            "ollama": _groq_status(),
        }
        return JSONResponse(response_payload)

    if field_id == "contact":
        question = field["fixed_question"]
        response_payload = {
            "done": False,
            "field": field_id,
            "label": field["label"],
            "question": question,
            "contact_type": True,
            "total_steps": len(CATEGORY_FLOW),
            "answered_steps": sum(1 for f in CATEGORY_FLOW if has_value(answers, f["id"])),
            "ollama": _groq_status(),
        }
        return JSONResponse(response_payload)

    step = generate_dynamic_step(answers, field_id)
    base_payload = {
        "done": False,
        "field": field_id,
        "label": field["label"],
        "total_steps": len(CATEGORY_FLOW),
        "answered_steps": sum(1 for f in CATEGORY_FLOW if has_value(answers, f["id"])),
        "ollama": _groq_status(),
    }

    if step.get("error"):
        return JSONResponse({**base_payload, "error": step["error"]})

    return JSONResponse(
        {
            **base_payload,
            "question": step["question"],
            "options": step["options"],
            "model_used": step.get("model"),
        }
    )


@app.post("/api/generate_assets")
async def api_generate_assets(request: Request):
    payload = await request.json()
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        answers = {}

    prompt = f"""
You are an expert ad copy assistant.
Use clear, punchy, and creative language.

Brief:
{json.dumps(answers, ensure_ascii=True)}

Generate exactly this format:

HEADLINES:
1.
2.
3.
4.
5.

TAGLINES:
1.
2.
3.
4.
5.

CTAS:
1.
2.
3.

IMAGE_PROMPTS:
1.
2.
3.

Rules:
- Headlines max 10 words each.
- Taglines max 12 words each.
- Image prompts must mention style, subject, and lighting.
- No extra explanation.
"""

    result = ask_llm(prompt)
    if not result["text"]:
        return JSONResponse(
            {
                "output": "",
                "error": result["error"],
                "ollama": _groq_status(),
            }
        )

    return JSONResponse(
        {
            "output": result["text"],
            "ollama": _groq_status(),
            "variant_engine": {
                "headlines": 5,
                "images": 3,
                "layouts": 2,
                "total_variants": 30,
                "layout_options": ["Clean product focus", "Bold text overlay"],
            },
        }
    )


@app.post("/api/generate_image")
async def api_generate_image(request: Request):
    payload = await request.json()
    prompt = payload.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    try:
        # Run imagegen.py as a subprocess (now in the same directory)
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "imagegen.py"), prompt],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"
        )
        
        output_lines = result.stdout.strip().split("\n")
        json_str = None
        for line in output_lines:
            if line.strip().startswith('{"images":'):
                json_str = line.strip()
                break
        
        if json_str is None:
            return JSONResponse({"error": "Invalid output format from generator"}, status_code=500)

        output_data = json.loads(json_str)
        return JSONResponse(output_data)

    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"Generation failed: {e.stderr}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/generate_video")
async def api_generate_video(request: Request):
    payload = await request.json()
    prompt = payload.get("prompt", "")
    image_path = payload.get("image_path", "")

    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    try:
        # Construct command (now in the same directory)
        cmd = [sys.executable, str(BASE_DIR / "videogen.py"), prompt]
        if image_path:
            # If image_path is absolute, use it; otherwise, relative to ROOT_DIR
            abs_img_path = str(ROOT_DIR / image_path) if not os.path.isabs(image_path) else image_path
            cmd.extend(["--images", abs_img_path])

        print(f"[VIDEO] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            cwd=str(ROOT_DIR) # Run from root where videogen.py expect libraries and saves files
        )

        # videogen.py saves to "final_ad_film.mp4" in its CWD.
        # We should move it to static/generated/ for serving.
        video_output_name = "final_ad_film.mp4"
        output_src = ROOT_DIR / video_output_name
        
        if not output_src.exists():
            print(f"[VIDEO] Error: Output file {output_src} not found. Stdout: {result.stdout}")
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
        print(f"[VIDEO] Script failed: {e.stderr}")
        return JSONResponse({"error": f"Video generation failed: {e.stderr}"}, status_code=500)
    except Exception as e:
        print(f"[VIDEO] Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
