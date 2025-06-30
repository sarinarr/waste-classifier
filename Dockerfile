# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install system dependencies (if needed by TensorFlow)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set Streamlit config port
EXPOSE 8501

# Start the app
CMD ["streamlit", "run", "Waste-Classifier.py", "--server.port=8501", "--server.enableCORS=false"]
