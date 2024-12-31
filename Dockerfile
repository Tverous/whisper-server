FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

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