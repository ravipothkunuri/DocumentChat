#!/bin/bash

# Start backend in background
python -m uvicorn backend.main:app --host localhost --port 8000 --log-level info &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend (this will run in foreground)
streamlit run frontend/app.py --server.port 5000 --server.address 0.0.0.0

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null
