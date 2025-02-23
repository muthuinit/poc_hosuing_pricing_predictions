# Use a base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the script and dataset
COPY house_prediction.py .

# Copy the service account key
COPY service-account-key.json /app/service-account-key.json

# Set the environment variable for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

# Set the entry point to run the script
ENTRYPOINT ["python", "house_prediction.py"]