# Whisper Speech-to-Text API

This API allows real-time and batch transcription of audio files using OpenAI's Whisper model. It supports both file-based and streaming audio input for transcription. The API exposes several endpoints for handling transcription jobs, whether they are triggered by URL, file upload, or streaming audio from a WebSocket.

## Table of Contents
- [API Overview](#api-overview)
- [Installation and Setup](#installation-and-setup)
- [Endpoints](#endpoints)
  - [Create a Job from Audio URL](#create-a-job-from-audio-url)
  - [Get Job Status](#get-job-status)
  - [Create a Job via File Upload](#create-a-job-via-file-upload)
  - [Real-Time Streaming Transcription](#real-time-streaming-transcription)
- [Examples](#examples)

---

## API Overview

This API supports:
1. **File-based transcription**: Upload or provide a URL for an audio file, and receive a transcription result.
2. **Real-time streaming transcription**: Stream audio via a WebSocket connection and get real-time transcription results.

The API uses [Whisper](https://github.com/openai/whisper) for transcription and [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection.

---

## Installation and Setup

To run the API locally, you need the following dependencies:
- Python 3.8+
- FastAPI
- Uvicorn
- Whisper (OpenAI's Whisper model)
- PyTorch
- Librosa
- Soundfile
- Aiohttp

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

To run the API:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

---

## Endpoints

### Create a Job from Audio URL

You can create a transcription job by providing an audio file's URL.

#### Request
- **Method**: `POST`
- **URL**: `/v1/jobs`
- **Body** (JSON):
  ```json
  {
    "source_config": {
      "url": "https://example.com/audio.mp3"
    },
    "metadata": "Optional metadata"
  }
  ```

#### Example using `curl`
```bash
curl -X 'POST' \
  'http://localhost:8000/v1/jobs' \
  -H 'Content-Type: application/json' \
  -d '{
    "source_config": {
      "url": "https://example.com/audio.mp3"
    },
    "metadata": "Audio metadata"
  }'
```

### Get Job Status

You can check the status of a transcription job using its `job_id`.

#### Request
- **Method**: `GET`
- **URL**: `/v1/jobs/{job_id}`

#### Example using `curl`
```bash
curl -X 'GET' \
  'http://localhost:8000/v1/jobs/<job_id>'
```

### Create a Job via File Upload

You can upload an audio file and create a transcription job.

#### Request
- **Method**: `POST`
- **URL**: `/v1/jobs/upload`
- **Body**: File upload
  - Use `multipart/form-data` to upload an audio file.

#### Example using `curl`
```bash
curl -X 'POST' \
  'http://localhost:8000/v1/jobs/upload' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/audio.mp3' \
  -F 'metadata="Optional metadata"'
```

### Real-Time Streaming Transcription

You can stream audio and get real-time transcriptions over a WebSocket connection.

#### WebSocket URL
- **URL**: `/v1/stream`

#### Example using `websocat`
```bash
websocat ws://localhost:8000/v1/stream
```

Once the connection is established, you can start sending audio chunks. The API will return real-time transcription data for each chunk.

---

## Examples

### 1. Create a Job from Audio URL

- **Request**:
  ```json
  {
    "source_config": {
      "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
    },
    "metadata": "Sample Audio File"
  }
  ```

- **`curl` Command**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/v1/jobs' \
    -H 'Content-Type: application/json' \
    -d '{
      "source_config": {
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
      },
      "metadata": "Sample Audio File"
    }'
  ```

### 2. Get Job Status

- **Request**: GET job status using `job_id`.
- **`curl` Command**:
  ```bash
  curl -X 'GET' \
    'http://localhost:8000/v1/jobs/abc123'
  ```

### 3. Create a Job via File Upload

- **Request**: Upload an audio file using `multipart/form-data`.
- **`curl` Command**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/v1/jobs/upload' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@/path/to/audio.mp3' \
    -F 'metadata="Test Audio File"'
  ```

### 4. Real-Time Streaming Transcription

1. Open a WebSocket connection:
   ```bash
   websocat ws://localhost:8000/v1/stream
   ```

2. Send audio chunks to the server (binary data) via the WebSocket connection.
3. Receive real-time transcription results.

---

## API Response Structure

### Create Job Response
- **Status**: `pending`, `in_progress`, `completed`, `error`
- **Job ID**: Unique ID for the job
- **Metadata**: Optional metadata

Example:
```json
{
  "id": "abc123",
  "created_on": "2024-12-31T00:00:00",
  "name": "audio.mp3",
  "status": "pending",
  "type": "async",
  "language": null,
  "metadata": "Sample metadata"
}
```

### Job Status Response
- **Status**: `pending`, `in_progress`, `completed`, `error`
- **Result**: Transcription result (if completed)
- **Error**: Error message (if failed)

Example:
```json
{
  "id": "abc123",
  "status": "completed",
  "result": {
    "text": "Hello, this is a test transcription.",
    "language": "en",
    "segments": []
  }
}
```

### Real-Time Streaming Response

- **Type**: `partial` (intermediate transcription) or `final` (finalized transcription)
- **Text**: The transcribed text

Example:
```json
{
  "type": "partial",
  "partial_text": "Hello, this is",
  "accumulated_text": "Hello, this is a test transcription."
}
```
