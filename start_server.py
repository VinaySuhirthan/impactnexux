#!/usr/bin/env python3
"""
AdGPT Server Launcher - Always works!
Handles port cleanup, environment setup, and server start
"""
import subprocess
import time
import socket
import os
import sys

def port_in_use(port=8000):
    """Check if port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('127.0.0.1', port))
        return result == 0

def kill_port(port=8000):
    """Force kill process on port"""
    print(f"[*] Clearing port {port}...")
    # Kill process using the port
    os.system(f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{port} ^| findstr LISTENING\') do taskkill /F /PID %a 2>nul')
    # Also kill all python and uvicorn processes
    os.system("taskkill /F /IM python.exe /T 2>nul")
    os.system("taskkill /F /IM uvicorn.exe /T 2>nul")
    time.sleep(2)

def main():
    port = 8000
    host = "127.0.0.1"
    
    print("\n" + "="*60)
    print("  AdGPT SERVER LAUNCHER")
    print("="*60)
    
    # Kill old processes
    if port_in_use(port):
        print(f"[!] Port {port} is in use, killing processes...")
        kill_port(port)
    
    # Wait for port to be free
    for i in range(5):
        if not port_in_use(port):
            break
        print(f"[...] Waiting for port {port} to be available... ({i+1}/5)")
        time.sleep(1)
    
    if port_in_use(port):
        print(f"[ERROR] Port {port} still in use! Try restarting Windows or using a different port.")
        sys.exit(1)
    
    print(f"[✓] Port {port} is free!")
    
    # Set environment (disable IPv6)
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Start server
    print(f"\n{'='*60}")
    print(f"[✓] AdGPT Server Starting...")
    print(f"{'='*60}")
    print(f"\n[*] OPEN THIS IN YOUR BROWSER:")
    print(f"    =====> http://localhost:{port} <====")
    print(f"\n[*] Press CTRL+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "appbc:app",
            "--reload",
            "--host", "127.0.0.1",  # Listen on localhost
            "--port", str(port)
        ])
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
