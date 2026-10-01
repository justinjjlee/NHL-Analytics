"""
Launcher for Rinklytics Streamlit Dashboard on Hugging Face Spaces and local environments.
"""
import os
import subprocess
import sys

# Required for Hugging Face ZeroGPU runtime initialization
try:
    import spaces

    @spaces.GPU
    def _init_gpu():
        return True

    _init_gpu()
except Exception:
    pass

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(root_dir, "app", "app.py")
    port = os.environ.get("PORT", "7860")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_file,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]
    subprocess.run(cmd)
