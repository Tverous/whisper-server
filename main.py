# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
import whisper
import numpy as np
import torch
import threading
from queue import Queue
import time
import asyncio
from typing import Optional, Dict, Any, Generator
import aiohttp
import io
from datetime import datetime
import uuid
import logging
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioConfig:
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_duration: float = 1.0,
                 vad_threshold: float = 0.3,
                 silence_duration: float = 0.3,
                 realtime_updates: bool = True,
                 update_interval: float = 0.2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.realtime_updates = realtime_updates
        self.update_interval = update_interval

class WhisperProcessor:
    def __init__(self, model_name: str = "base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name).to(self.device)
        
        # Load VAD model
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=False
        )
        self.vad_model = self.vad_model.to(self.device)
        self.get_speech_timestamps = utils[0]
        
    def transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous transcription using Whisper."""
        try:
            # Convert audio to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Normalize audio if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0
                
            # Get transcription
            result = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=False if self.device == "cpu" else True
            )
            
            # Format segments
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "words": []  # Whisper doesn't provide word-level timestamps by default
                })
            
            return {
                "segments": segments,
                "language": result["language"],
                "text": result["text"]
            }
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            raise
            
    def check_voice_activity(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio chunk contains voice activity."""
        try:
            audio_tensor = torch.from_numpy(audio_data).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_tensor = audio_tensor.to(self.device)
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=0.5,
                sampling_rate=sample_rate
            )
            
            return len(speech_timestamps) > 0
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {str(e)}")
            return False

class StreamingSession:
    def __init__(self, processor: WhisperProcessor, config: AudioConfig):
        self.processor = processor
        self.config = config
        self.buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.last_voice_timestamp = 0
        self.min_samples_for_processing = int(self.config.sample_rate * 2)  # At least 2 seconds of audio
        
    def get_buffer_length(self) -> int:
        """Get the current buffer length in samples."""
        return len(self.buffer) if self.buffer is not None else 0
        
    async def process_chunk(self, chunk: bytes) -> Dict[str, Any]:
        """Process an incoming audio chunk."""
        try:
            # Handle end of stream
            if not chunk:
                if self.get_buffer_length() > 0:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.processor.transcribe_sync,
                        self.buffer
                    )
                    self.buffer = np.array([], dtype=np.float32)
                    return {
                        "type": "final",
                        **result,
                        "is_speaking": False
                    }
                return {"type": "final", "text": "", "is_speaking": False}
            
            # Convert bytes to numpy array and normalize
            audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
            
            # Always accumulate the buffer
            if self.buffer is None:
                self.buffer = audio_data
            else:
                self.buffer = np.concatenate([self.buffer, audio_data])
            
            # Log buffer size for debugging
            logger.debug(f"Current buffer size: {self.get_buffer_length()} samples")
            
            # Process buffer if it's large enough
            if self.get_buffer_length() >= self.min_samples_for_processing:
                logger.info(f"Processing buffer of size {self.get_buffer_length()}")
                
                # Process current buffer
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.processor.transcribe_sync,
                    self.buffer
                )
                
                # Clear buffer
                self.buffer = np.array([], dtype=np.float32)
                
                return {
                    "type": "final",
                    **result,
                    "is_speaking": True
                }
            
            # Return buffering status
            return {
                "type": "buffering",
                "is_speaking": True,
                "buffer_size": self.get_buffer_length()
            }
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise
            
    async def end_session(self) -> Optional[Dict[str, Any]]:
        """End the streaming session and process any remaining audio."""
        try:
            if self.get_buffer_length() > 0:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.processor.transcribe_sync,
                    self.buffer
                )
                self.buffer = np.array([], dtype=np.float32)
                return {
                    "type": "final",
                    **result,
                    "is_speaking": False
                }
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            return {
                "type": "error",
                "error": str(e),
                "is_speaking": False
            }
        return None

# API Models
class SourceConfig(BaseModel):
    url: HttpUrl

class TranscriptionRequest(BaseModel):
    source_config: SourceConfig
    metadata: Optional[str] = None

class TranscriptionJob(BaseModel):
    id: str
    created_on: datetime
    name: str
    status: str
    type: str
    language: Optional[str]
    metadata: Optional[str]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(title="Whisper Speech-to-Text API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper processor
whisper_processor = WhisperProcessor("base")

# In-memory job storage
jobs: Dict[str, TranscriptionJob] = {}

async def download_audio(url: HttpUrl) -> bytes:
    """Download audio file from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url)) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download audio file")
            return await response.read()

async def process_transcription(job_id: str, audio_data: bytes):
    """Process transcription in background."""
    try:
        jobs[job_id].status = "in_progress"
        
        # Convert audio data to numpy array
        audio_array = sf.read(io.BytesIO(audio_data))[0]
        
        # Process using Whisper
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            whisper_processor.transcribe_sync,
            audio_array
        )
        
        jobs[job_id].status = "completed"
        jobs[job_id].result = result
        jobs[job_id].language = result.get("language")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        jobs[job_id].status = "error"
        jobs[job_id].error = str(e)

@app.post("/v1/jobs")
async def create_transcription_job_url(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """Create a new async transcription job."""
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = TranscriptionJob(
        id=job_id,
        created_on=datetime.utcnow(),
        name=request.source_config.url.split("/")[-1],
        status="pending",
        type="async",
        language=None,
        metadata=request.metadata
    )
    jobs[job_id] = job
    
    try:
        # Download audio
        audio_data = await download_audio(request.source_config.url)
        
        # Process transcription in background
        background_tasks.add_task(process_transcription, job_id, audio_data)
        
        return job
        
    except Exception as e:
        jobs[job_id].status = "error"
        jobs[job_id].error = str(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a transcription job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/v1/jobs/upload")
async def create_transcription_job_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Create a new async transcription job using file upload."""
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = TranscriptionJob(
        id=job_id,
        created_on=datetime.utcnow(),
        name=file.filename,
        status="pending",
        type="async",
        language=None,
        metadata=metadata
    )
    jobs[job_id] = job
    
    try:
        # Read file content
        audio_data = await file.read()
        
        # Process transcription in background
        background_tasks.add_task(process_transcription, job_id, audio_data)
        
        return job
        
    except Exception as e:
        jobs[job_id].status = "error"
        jobs[job_id].error = str(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time streaming transcription endpoint."""
    await websocket.accept()
    
    session = None
    try:
        # Initialize streaming session with appropriate config
        config = AudioConfig(
            chunk_duration=5.0,  # 5 seconds chunks
            silence_duration=0.5,  # 0.5 seconds of silence to mark end of speech
            realtime_updates=True,
            update_interval=0.2
        )
        session = StreamingSession(whisper_processor, config)
        
        while True:
            try:
                # Receive audio chunk with timeout
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                
                if not data or len(data) == 0:
                    logger.info("Received end of stream signal")
                    # Process any remaining audio
                    final_result = await session.process_chunk(b"")
                    if final_result:
                        await websocket.send_json(final_result)
                    break
                    
                # Process chunk
                try:
                    result = await session.process_chunk(data)
                    
                    print(f"Result: {result}")
                    
                    if result:
                        await websocket.send_json(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
                    break
                
            except asyncio.TimeoutError:
                logger.debug("Websocket receive timeout - continuing")
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed by client")
                break
            except Exception as e:
                logger.error(f"Error in websocket communication: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
                except:
                    pass
                break
                
    except Exception as e:
        logger.error(f"Websocket error: {str(e)}")
    
    finally:
        if session:
            try:
                final_result = await session.end_session()
                if final_result:
                    await websocket.send_json(final_result)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)