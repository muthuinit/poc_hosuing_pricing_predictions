FROM gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-7:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the prediction script
COPY house_prediction.py .

# Ensure the service account is used without a JSON key
# Assuming the environment already has proper permissions
ENV GOOGLE_APPLICATION_CREDENTIALS=""

# Run the script
ENTRYPOINT ["python", "house_prediction.py"]
