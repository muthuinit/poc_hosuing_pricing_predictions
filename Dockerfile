# Use a base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Show Python and package versions for debugging
RUN python --version && pip freeze

# Copy the script
COPY house_prediction.py .

# Set the entry point to run the script
ENTRYPOINT ["python", "house_prediction.py"]