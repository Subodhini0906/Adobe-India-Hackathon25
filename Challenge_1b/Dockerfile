FROM python:3.11-slim

WORKDIR /app

# Install system deps first
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip with retries
RUN python -m pip install --upgrade pip \
    --timeout=100 \
    --retries 10 \
    --default-timeout=100

# Install requirements with retries
COPY requirements.txt .
RUN pip install \
    --no-cache-dir \
    --timeout=100 \
    --retries 10 \
    --default-timeout=100 \
    -r requirements.txt

COPY . .

CMD ["python", "main.py"]