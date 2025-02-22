# Use a base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the training script and dataset
COPY house_prediction.py .
COPY Housing.csv .

# Set the entry point for the container
ENTRYPOINT ["python", "house_prediction.py"]