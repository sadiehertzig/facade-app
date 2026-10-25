"""Shared pytest setup for the server tests.

The worker modules import each other by file name (they are run as scripts from
server/worker/), and the demo is the `photoguard_demo` package under server/.
Put both directories on sys.path so every test imports them the same way the
server does. Nothing here needs a GPU, model weights, gradio or the network.
"""
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
REPO = SERVER.parent
for path in (SERVER / "worker", SERVER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
