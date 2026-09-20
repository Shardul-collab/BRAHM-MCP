import sys
from pathlib import Path

# VIDUR's modules import each other by bare top-level name ("from parsers import
# xrd"), exactly as mcp_server.py sets things up with sys.path.insert(0, agents/vidur).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
