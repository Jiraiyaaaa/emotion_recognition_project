# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN set -eux; \
    for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git && \
        rm -rf /var/lib/apt/lists/* && break || sleep 10; \
    done

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --timeout 180 --retries 10 -r requirements.txt

# Copy project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the app
ENV PORT=5000
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
