FROM python:3.11-slim

WORKDIR /app/backend

# Create data directory for SQLite persistence
RUN mkdir -p /data

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY backend/ .

ENV PYTHONPATH=/app/backend \
    DB_PATH=/data/airsync.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
