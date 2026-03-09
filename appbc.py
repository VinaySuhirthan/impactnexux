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

# ── Gemini configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBXT8nbnApafaYSYa-Cgxv8QJV3pfZfWeE")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

# ── State that mirrors app.py's "ollama" state (renamed to "gemini") ──────────
GEMINI_ENABLED = (
    os.getenv("GEMINI_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
)
GEMINI_DISABLED_UNTIL = 0.0
GEMINI_LAST_ERROR = ""
ACTIVE_MODEL = GEMINI_MODEL
GEMINI_RETRY_SECONDS = int(os.getenv("GEMINI_RETRY_SECONDS", "5"))

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


# ── Gemini error state helpers ────────────────────────────────────────────────

def _set_gemini_error(message: str, cooldown_seconds: Optional[int] = None) -> None:
    global GEMINI_DISABLED_UNTIL, GEMINI_LAST_ERROR
    GEMINI_LAST_ERROR = message
    if cooldown_seconds is not None:
        GEMINI_DISABLED_UNTIL = time.time() + max(1, cooldown_seconds)


def _clear_gemini_error() -> None:
    global GEMINI_DISABLED_UNTIL, GEMINI_LAST_ERROR
    GEMINI_DISABLED_UNTIL = 0.0
    GEMINI_LAST_ERROR = ""


def _gemini_status() -> Dict[str, Any]:
    retry_after = 0
    if GEMINI_DISABLED_UNTIL > 0:
        retry_after = max(0, int(round(GEMINI_DISABLED_UNTIL - time.time())))

    return {
        "enabled": GEMINI_ENABLED,
        "configured_model": GEMINI_MODEL,
        "active_model": ACTIVE_MODEL,
        "last_error": GEMINI_LAST_ERROR,
        "retry_after_seconds": retry_after,
    }


# ── Core LLM call ─────────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> Dict[str, Any]:
    global ACTIVE_MODEL

    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("AIzaSy") == False:
        return {
            "text": "",
            "model": ACTIVE_MODEL,
            "error": "GEMINI_API_KEY is not set or invalid. Please set GEMINI_API_KEY environment variable with a valid key.",
        }

    if not GEMINI_ENABLED:
        return {
            "text": "",
            "model": ACTIVE_MODEL,
            "error": "Gemini is disabled. Set GEMINI_ENABLED=true to use dynamic questions.",
        }

    now = time.time()
    if now < GEMINI_DISABLED_UNTIL:
        wait_seconds = max(1, int(round(GEMINI_DISABLED_UNTIL - now)))
        message = GEMINI_LAST_ERROR or f"Gemini is cooling down. Retry in {wait_seconds}s."
        return {"text": "", "model": ACTIVE_MODEL, "error": message}

    try:
        print(f"[BRAIN:GEMINI] Sending REST prompt to '{GEMINI_MODEL}' ({len(prompt)} chars)")
        print(f"[BRAIN:GEMINI] Prompt preview: {prompt[:120].strip()!r}...")
        print(f"[BRAIN:GEMINI] API Key check: {GEMINI_API_KEY[:20]}...")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        
        print(f"[BRAIN:GEMINI] Requesting: {url[:80]}...")
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        print(f"[BRAIN:GEMINI] Status code: {resp.status_code}")
        print(f"[BRAIN:GEMINI] Response: {resp.text[:500]}")
        
        resp.raise_for_status()
        data = resp.json()
        
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = text.strip() if text else ""
        except (KeyError, IndexError) as e:
            print(f"[BRAIN:GEMINI] Parse error: {e}, data: {data}")
            raise ValueError(f"Invalid block format from Gemini output: {e}")
            
        if not text:
            raise ValueError("Gemini returned an empty response.")
        ACTIVE_MODEL = GEMINI_MODEL
        _clear_gemini_error()
        print(f"[BRAIN:GEMINI] Response received ({len(text)} chars): {text[:120].strip()!r}...")
        return {"text": text, "model": GEMINI_MODEL, "error": ""}
    except requests.HTTPError as exc:
        message = f"Gemini HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        print(f"[BRAIN:GEMINI] HTTP Error: {message}")
        _set_gemini_error(message, cooldown_seconds=GEMINI_RETRY_SECONDS)
        return {"text": "", "model": ACTIVE_MODEL, "error": message}
    except Exception as exc:
        import traceback
        error_trace = traceback.format_exc()
        message = f"Gemini error: {exc}"
        print(f"[BRAIN:GEMINI] Exception: {error_trace}")
        _set_gemini_error(message, cooldown_seconds=GEMINI_RETRY_SECONDS)
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
    # Strip markdown code fences Gemini sometimes wraps around JSON
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

@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/api/ollama_status")
def api_ollama_status():
    """Kept as /api/ollama_status so the frontend doesn't need changes."""
    return JSONResponse(_gemini_status())


@app.post("/api/next_question")
async def api_next_question(request: Request):
    payload = await request.json()
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        answers = {}

    field_id = next_field_id(answers)
    if not field_id:
        return JSONResponse({"done": True, "ollama": _gemini_status()})

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
            "ollama": _gemini_status(),
        }
        return JSONResponse(response_payload)

    step = generate_dynamic_step(answers, field_id)
    base_payload = {
        "done": False,
        "field": field_id,
        "label": field["label"],
        "total_steps": len(CATEGORY_FLOW),
        "answered_steps": sum(1 for f in CATEGORY_FLOW if has_value(answers, f["id"])),
        "ollama": _gemini_status(),
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
                "ollama": _gemini_status(),
            }
        )

    return JSONResponse(
        {
            "output": result["text"],
            "ollama": _gemini_status(),
            "variant_engine": {
                "headlines": 5,
                "images": 3,
                "layouts": 2,
                "total_variants": 30,
                "layout_options": ["Clean product focus", "Bold text overlay"],
            },
        }
    )
