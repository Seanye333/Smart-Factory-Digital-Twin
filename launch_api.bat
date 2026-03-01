@echo off
echo Starting Smart Factory Digital Twin API on http://localhost:8000 ...
echo Swagger UI: http://localhost:8000/docs
"C:\Users\seany\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn digital_twin.api.server:app --host 0.0.0.0 --port 8000 --reload
pause
