# Use a base image with Python 3.9
FROM python:3.9-slim

# Set environment variables to avoid Python buffer issues
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for libraries like xgboost)
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Double-check xgboost is installed
RUN pip show xgboost

# Show Python and package versions for debugging
RUN python --version && pip freeze

# Copy the script
COPY house_prediction.py .

# Set the entry point to run the script
ENTRYPOINT ["python", "house_prediction.py"]
