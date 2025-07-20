# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential ffmpeg libsm6 libxext6 git portaudio19-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
