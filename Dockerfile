
# Same Base as the VLLM Worker
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    wget \
    libavif-bin \
    exiftool \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies with pip
# For uv alternative, see Dockerfile.uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Run the Handler
CMD python -u /app/handler.py


