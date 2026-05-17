FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serving/ /app/serving/
COPY models/ /app/models/
COPY training/ /app/training/
COPY monitoring/ /app/monitoring/
COPY storage/ /app/storage/
COPY ingestion/ /app/ingestion/
COPY features/ /app/features/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
