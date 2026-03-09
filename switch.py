brain = True  # True = Qwen / Ollama (app.py) | False = Groq (appbc:app)

import subprocess
import sys

module = "app:app" if brain else "appbc:app"
label = "Qwen / Ollama" if brain else "Groq"

print(f"[switch] Brain: {label}")
print(f"[switch] Module: {module}")
print("[switch] Starting server on http://127.0.0.1:8000")

proc = subprocess.Popen([
    sys.executable, "-m", "uvicorn", module,
    "--reload",
    "--host", "127.0.0.1",
    "--port", "8000",
])

try:
    proc.wait()
except KeyboardInterrupt:
    print("[switch] Shutting down server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()