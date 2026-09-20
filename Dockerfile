from python:3.11-slim

workdir /app

copy requirements.txt .
run pip install --no-cache-dir -r requirements.txt

copy app/ ./app/
copy artifacts/ ./artifacts/

expose 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}