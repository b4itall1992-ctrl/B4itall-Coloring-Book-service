# Dockerfile – stable on Railway
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Minimal system libs OpenCV needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway sets PORT; uvicorn will use it
ENV PORT=8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
