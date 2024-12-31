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
                 chunk_duration: float = 0.5,  # 0.5 seconds per chunk
                 vad_threshold: float = 0.3,
                 silence_duration: float = 0.3,
                 realtime_updates: bool = True,
                 update_interval: float = 0.1):  # Reduced to handle shorter chunks
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
                
            # Get transcription with additional parameters for better accuracy
            result = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=False if self.device == "cpu" else True,
                best_of=5,  # Increase beam search paths
                beam_size=5,  # Increase beam size
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                temperature=0.0  # Reduce randomness in output
            )
            
            # Format segments with confidence scores
            segments = []
            for segment in result["segments"]:
                avg_logprob = segment.get("avg_logprob", -1)
                no_speech_prob = segment.get("no_speech_prob", 0)
                
                # Calculate confidence score
                confidence = 1.0 + avg_logprob  # Convert log prob to probability
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                confidence *= (1.0 - no_speech_prob)  # Reduce confidence if high no_speech_prob
                
                segments.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": confidence,
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech_prob
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

# StreamingSession class modifications
class StreamingSession:
    def __init__(self, processor: WhisperProcessor, config: AudioConfig):
        self.processor = processor
        self.config = config
        
        # Audio buffer for current chunk accumulation
        self.buffer = np.array([], dtype=np.float32)
        
        # State tracking
        self.is_speaking = False
        self.last_voice_timestamp = 0
        
        # --- Buffer size thresholds ---
        self.MIN_CHUNK_SIZE = int(0.5 * self.config.sample_rate)   # 0.5 s
        self.OPTIMAL_CHUNK_SIZE = int(2.0 * self.config.sample_rate)  # 2 s
        self.MAX_CHUNK_SIZE = int(10.0 * self.config.sample_rate)     # 10 s
        
        # Confidence thresholds
        self.MIN_CONFIDENCE_THRESHOLD = 0.6
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        
        # Tracks the **highest-confidence** partial result so far
        self.last_result = None
        self.last_confidence = 0.0
        
        # Fully accumulated text and segments across multiple chunks
        self.accumulated_text = ""
        self.accumulated_segments = []
    
    def get_buffer_length(self) -> int:
        """Get current buffer length in samples."""
        return len(self.buffer)

    def calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on multiple factors."""
        try:
            # Extract segment confidences from Whisper results
            segment_confidences = []
            for segment in result.get("segments", []):
                avg_logprob = segment.get("avg_logprob", -1)
                no_speech_prob = segment.get("no_speech_prob", 0)
                
                # Convert log prob to probability
                conf = np.exp(avg_logprob)
                conf *= (1.0 - no_speech_prob)
                segment_confidences.append(conf)

            # Average across segments
            avg_confidence = (
                sum(segment_confidences) / len(segment_confidences)
                if segment_confidences else 0.0
            )
            
            # Text length factor
            text = result.get("text", "").strip()
            text_length_factor = min(len(text.split()) / 3, 1.0)
            
            # Simple penalties
            has_ellipsis = "..." in text
            has_question_marks = "???" in text
            has_repeated_chars = any(c * 3 in text for c in text)
            penalties = sum([
                0.2 if has_ellipsis else 0,
                0.2 if has_question_marks else 0,
                0.2 if has_repeated_chars else 0
            ])
            
            # Combine
            confidence = (avg_confidence * 0.8 + text_length_factor * 0.2) * (1.0 - penalties)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    async def process_chunk(self, chunk: bytes) -> Dict[str, Any]:
        """
        Process an incoming audio chunk:
          1) Accumulate audio in a buffer.
          2) Check if we should run transcription (size/time-based or silence).
          3) Compare new transcription confidence to previous:
             - If new is higher, replace the old.
             - If old is higher, keep old.
          4) If voice ended, finalize text and reset.
        """
        try:
            # Handle end of stream
            if not chunk:
                # If there's leftover audio, transcribe it
                if self.get_buffer_length() >= self.MIN_CHUNK_SIZE:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.processor.transcribe_sync,
                        self.buffer
                    )
                    # This final chunk result is likely high confidence 
                    return self._prepare_final_result(result, finalize=True)
                
                # If too small leftover, finalize with whatever we have
                return {
                    "type": "final",
                    "text": self.accumulated_text,
                    "segments": self.accumulated_segments,
                    "is_speaking": False
                }

            # Convert chunk to float32 in [-1,1]
            audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data /= 32768.0

            # Check voice activity
            has_speech = self.processor.check_voice_activity(audio_data)
            current_time = time.time()

            # Update speaking state
            if has_speech:
                self.is_speaking = True
                self.last_voice_timestamp = current_time
            else:
                # If silence is enough to consider speech ended
                if (current_time - self.last_voice_timestamp) > self.config.silence_duration:
                    self.is_speaking = False
            
            # Accumulate data
            self.buffer = np.concatenate([self.buffer, audio_data]) \
                          if len(self.buffer) > 0 else audio_data
            
            current_buffer_length = self.get_buffer_length()

            # Condition for a forced transcription
            should_process = (
                current_buffer_length >= self.MAX_CHUNK_SIZE or
                (current_buffer_length >= self.OPTIMAL_CHUNK_SIZE and not self.is_speaking) or
                (current_buffer_length >= self.MIN_CHUNK_SIZE and
                 not self.is_speaking and
                 (current_time - self.last_voice_timestamp) > self.config.silence_duration * 2)
            )

            if should_process:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.processor.transcribe_sync,
                    self.buffer
                )
                current_confidence = self.calculate_confidence(result)
                logger.info(f"New transcription confidence: {current_confidence:.2f}")

                # Compare vs. previously stored best result
                if self.last_result:
                    if current_confidence > self.last_confidence:
                        # Replace with better new result
                        logger.info("New chunk is more confident, replacing old partial result.")
                        self.last_result = result
                        self.last_confidence = current_confidence
                    else:
                        # Keep old
                        logger.info("Old chunk result had higher confidence, keeping it.")
                        result = self.last_result  # we revert to old
                else:
                    # No previous result => store new
                    self.last_result = result
                    self.last_confidence = current_confidence

                # If confidence is too low, keep buffering unless we reached max
                if (current_confidence < self.MIN_CONFIDENCE_THRESHOLD
                        and current_buffer_length < self.MAX_CHUNK_SIZE):
                    return {
                        "type": "buffering",
                        "is_speaking": self.is_speaking,
                        "buffer_size": current_buffer_length,
                        "confidence": current_confidence
                    }
                
                # Voice ended => finalize
                if not self.is_speaking:
                    final_data = self._prepare_final_result(self.last_result, finalize=True)
                    self._reset_streaming_state()  # Clear buffers & state
                    return final_data
                else:
                    # We are still speaking => partial
                    # Update accumulated text with the best partial
                    self._update_accumulated_text(self.last_result, self.last_confidence)
                    
                    # Reset buffer after we used it for partial
                    self.buffer = np.array([], dtype=np.float32)
                    
                    return self._prepare_partial_result(self.last_result)

            # Otherwise, keep buffering
            return {
                "type": "buffering",
                "is_speaking": self.is_speaking,
                "buffer_size": current_buffer_length
            }

        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise

    def _reset_streaming_state(self):
        """Reset entire streaming state after finalizing or stopping."""
        self.buffer = np.array([], dtype=np.float32)
        self.last_result = None
        self.last_confidence = 0.0
        self.is_speaking = False

    def _update_accumulated_text(self, result: Dict[str, Any], confidence: float) -> None:
        """Add the new text to our global accumulation if it is not redundant."""
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        
        # For high confidence, add it directly
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            self.accumulated_text = (self.accumulated_text + " " + text).strip()
            self.accumulated_segments.extend(segments)
        else:
            # For lower confidence, check redundancy to avoid spamming
            if text and not self._is_redundant_text(text):
                self.accumulated_text = (self.accumulated_text + " " + text).strip()
                self.accumulated_segments.extend(segments)

    def _is_redundant_text(self, new_text: str) -> bool:
        """Check if new_text is mostly overlapping with accumulated_text."""
        new_words = set(new_text.lower().split())
        accumulated_words = set(self.accumulated_text.lower().split())
        overlap = len(new_words.intersection(accumulated_words))
        # If >70% overlap => redundant
        return (len(new_words) > 0 and overlap / len(new_words) > 0.7)
    
    def _prepare_partial_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a partial result structure after each chunk if still speaking."""
        return {
            "type": "partial",
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language"),
            "is_speaking": self.is_speaking,
            "confidence": self.last_confidence
        }

    def _prepare_final_result(self, result: Dict[str, Any], finalize=False) -> Dict[str, Any]:
        """Prepare a final result message, optionally finalizing all text."""
        if finalize:
            # Merge last chunk's text into the accumulated text one last time
            if result:
                self._update_accumulated_text(result, self.last_confidence)
            
            return {
                "type": "final",
                "text": self.accumulated_text,
                "segments": self.accumulated_segments,
                "language": result.get("language") if result else None,
                "is_speaking": False,
                "confidence": self.last_confidence
            }
        else:
            # Return the chunk's final info but not finalize the entire conversation
            return {
                "type": "partial",
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language"),
                "is_speaking": self.is_speaking,
                "confidence": self.last_confidence
            }

    async def end_session(self) -> Dict[str, Any]:
        """
        Called at websocket close or whenever we want to finalize
        any leftover data in the buffer.
        """
        if self.get_buffer_length() >= self.MIN_CHUNK_SIZE:
            # Transcribe leftover
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.processor.transcribe_sync,
                self.buffer
            )
            return self._prepare_final_result(result, finalize=True)
        else:
            # Nothing leftover or too small
            return {
                "type": "final",
                "text": self.accumulated_text,
                "segments": self.accumulated_segments,
                "is_speaking": False
            }


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
        # Initialize streaming session with 0.5s chunk configuration
        config = AudioConfig(
            chunk_duration=0.5,     # 0.5 seconds per chunk
            silence_duration=0.3,    # 0.3 seconds of silence to mark end of speech
            realtime_updates=True,
            update_interval=0.1      # Reduced interval for more responsive updates
        )
        session = StreamingSession(whisper_processor, config)
        
        while True:
            try:
                # Receive audio chunk with shorter timeout
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                
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
                    if result:
                        await websocket.send_json(result)
                        
                        # Log final results for debugging
                        if result.get("type") == "final":
                            logger.info(f"Transcription: {result.get('text', '')}")
                            
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