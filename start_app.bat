@echo off
echo Starting Jalandhar Sentinel AI...
start cmd /k "cd backend && python app.py"
start cmd /k "cd frontend && npm run dev"
echo Both servers have been started in new windows!
echo Please open http://localhost:5173 once Vite is ready.
