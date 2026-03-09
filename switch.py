brain = False  # True = Qwen / Ollama (app.py) | False = Gemini (appbc.py)

import subprocess
import sys

module = "app:app" if brain else "appbc:app"
label = "Qwen / Ollama" if brain else "Gemini"

print(f"[switch] Brain: {label}")
print(f"[switch] Module: {module}")
print("[switch] Starting server on http://127.0.0.1:8000")

subprocess.run([
    sys.executable, "-m", "uvicorn", module,
    "--reload",
    "--host", "127.0.0.1",
    "--port", "8000",
])