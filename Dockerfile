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
COPY Housing.csv .

# Command to run the script
CMD ["python", "house_prediction.py", "--data_path", "Housing.csv", "--model_dir", "gs://your-bucket-name/path/to/model"]