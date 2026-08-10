#!/bin/bash
cd /mnt/d/SQL_IMP_AI_Project
source venv/bin/activate 2>/dev/null || true
python -m uvicorn api:app --host 0.0.0.0 --port 8000
exec bash
