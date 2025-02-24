# Use a base image with Python 3.9
FROM python:3.9-slim

# Set environment variables to avoid Python buffer issues
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies (if needed for xgboost or GCP libraries)
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Show Python and package versions for debugging
RUN python --version && pip freeze

# Copy the script
COPY house_prediction.py .

# Ensure the Google credentials are set (optional, based on deployment)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/your-service-account-key.json

# Set the entry point to run the script
ENTRYPOINT ["python", "house_prediction.py"]
