#!/usr/bin/env python3
"""
Script to start the Atemporal FastAPI backend for development/testing.

Usage:
    python scripts/start_backend.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    
    # Check if backend directory exists
    if not backend_dir.exists():
        print(f"Error: Backend directory not found at {backend_dir}")
        sys.exit(1)
    
    # Check if main.py exists
    main_py = backend_dir / "main.py"
    if not main_py.exists():
        print(f"Error: main.py not found at {main_py}")
        sys.exit(1)
    
    print("Starting Atemporal FastAPI backend...")
    print(f"Backend directory: {backend_dir}")
    print("Server will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Change to backend directory and run uvicorn
        os.chdir(backend_dir)
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\nBackend server stopped.")
    except FileNotFoundError:
        print("Error: uvicorn not found. Please install dependencies with 'uv sync' first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
