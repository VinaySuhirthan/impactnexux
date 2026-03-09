# AdGPT

A FastAPI-based web application for generating ad copy using either Ollama (offline) or Gemini AI.

## Setup

1. Ensure Python 3.8+ is installed.
2. Clone or download the project.
3. Run the setup:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

4. For Gemini: Copy `.env.example` to `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

5. For Ollama: Ensure Ollama is installed and running locally.

## Running the Application

### Using Gemini (default)
Simply run:
```bash
run.bat
```

### Switching to Ollama/Qwen
Edit `switch.py` and set `brain = True`, then run:
```bash
python switch.py
```

Or manually:
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Switching to Gemini
Edit `switch.py` and set `brain = False`, then run:
```bash
python switch.py
```

Or manually:
```bash
python -m uvicorn appbc:app --reload --host 127.0.0.1 --port 8000
```

This will activate the virtual environment and start the server on http://localhost:8000

## Features

- Dynamic question generation using Ollama (offline) or Gemini AI
- Ad copy generation
- Web interface

## Troubleshooting

If the server doesn't start:
- Ensure no other process is using port 8000
- Restart your computer if issues persist
- For Gemini: Check that GEMINI_API_KEY is set in .env
- For Ollama: Ensure Ollama service is running