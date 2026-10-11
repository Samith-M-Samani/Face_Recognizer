"""
Launcher script for Face Recognition ML Studio.
Run:
    py app.py
Opens http://127.0.0.1:8000 in your browser.
"""

import sys
import webbrowser
import uvicorn
from server import app

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print("=" * 65)
    print("      FACE RECOGNITION ML STUDIO (MANUAL KNN & METRICS)")
    print("=" * 65)
    print(f"[*] Starting local server on: {url}")
    print("[*] Serving interactive evaluation studio and real-time recognizer.")
    print("[*] Press Ctrl+C in terminal to stop.")
    print("=" * 65)
    
    # Try opening browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("server:app", host=host, port=port, reload=False, log_level="info")
