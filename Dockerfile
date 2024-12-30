FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir --upgrade

# Create model cache directory
RUN mkdir -p /app/models

# Copy source code
COPY . .

# CMD ["stt-server", "-r", "large-v2", "--enable_realtime_transcription", "--debug"]
CMD ["python", "faster.py"]