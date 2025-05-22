FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for textract and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    antiword \
    unrtf \
    tesseract-ocr \
    libjpeg-dev \
    swig \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch with CUDA support (if needed)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p input output cv_dummy models

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]