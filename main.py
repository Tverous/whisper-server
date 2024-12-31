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
                condition_on_previous_text=False,
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
class StreamingSession:
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        
        # -------------------------
        # SLIDING WINDOW SETTINGS
        # -------------------------
        self.SLIDING_WINDOW_SEC = 10.0  # total window size in seconds
        self.MIN_TRANSCIBE_WINDOW_SEC = 1.0  # min audio needed before we transcribe
        self.sample_rate = self.config.sample_rate
        
        # Convert durations to # of samples
        self.SLIDING_WINDOW_SAMPLES = int(self.SLIDING_WINDOW_SEC * self.sample_rate)
        self.MIN_TRANSCIBE_WINDOW_SAMPLES = int(self.MIN_TRANSCIBE_WINDOW_SEC * self.sample_rate)
        
        # Rolling buffer that always holds last ~3s of audio
        self.sliding_window_buffer = np.array([], dtype=np.float32)
        
        # For voice activity detection
        self.is_speaking = False
        self.last_voice_timestamp = 0
        
        # Confidence thresholds
        self.MIN_CONFIDENCE_THRESHOLD = 0.6
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        
        # Best partial result from the current utterance
        self.last_partial_text = ""
        self.last_partial_confidence = 0.0
        
        # Final, accumulated transcript across all utterances
        self.accumulated_text = ""
        
        # States for controlling finalization
        self.currently_in_utterance = False
    
    def _append_to_sliding_window(self, audio_data: np.ndarray):
        """
        Add new samples to the sliding window and trim if over capacity.
        """
        # Append
        if len(self.sliding_window_buffer) == 0:
            self.sliding_window_buffer = audio_data
        else:
            self.sliding_window_buffer = np.concatenate([self.sliding_window_buffer, audio_data])
        
        # Trim if we exceed the target window size
        overflow = len(self.sliding_window_buffer) - self.SLIDING_WINDOW_SAMPLES
        if overflow > 0:
            self.sliding_window_buffer = self.sliding_window_buffer[overflow:]
    
    async def process_chunk(self, chunk: bytes) -> Dict[str, Any]:
        """
        Processes the latest chunk by:
          1) Converting to float32 and normalizing
          2) Checking voice activity
          3) Appending to the sliding window
          4) Possibly transcribing the last ~3s
          5) Doing a simple alignment with previous partial
        """
        try:
            # If chunk is empty => end of stream
            if not chunk:
                logger.info("Received end-of-stream, finalizing.")
                return await self._finalize_transcription()
            
            # Convert chunk to float32 [-1, 1]
            new_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if np.abs(new_data).max() > 1.0:
                new_data /= 32768.0
            
            # Check VAD
            has_speech = self.processor.check_voice_activity(new_data)
            current_time = time.time()
            
            if has_speech:
                self.is_speaking = True
                self.last_voice_timestamp = current_time
                self.currently_in_utterance = True
            else:
                # If silent for > silence_duration, consider user has stopped talking
                if (current_time - self.last_voice_timestamp) > self.config.silence_duration:
                    self.is_speaking = False
            
            # Append to the sliding window
            self._append_to_sliding_window(new_data)
            
            # Decide if we should run transcription
            # We only transcribe if we have at least a min length & (time-based or event-based)
            window_len = len(self.sliding_window_buffer)
            if window_len >= self.MIN_TRANSCIBE_WINDOW_SAMPLES:
                
                # For demonstration, we will transcribe:
                #   1) if the user is not speaking now AND we have enough silence
                #   2) or if the window is "full"
                #   3) or if we haven't transcribed in a while
                should_transcribe = False
                
                # Condition 1: user just went silent
                if not self.is_speaking and self.currently_in_utterance:
                    # If user was speaking but now is silent for a certain period,
                    # let's transcribe the last window
                    quiet_time = current_time - self.last_voice_timestamp
                    if quiet_time >= (self.config.silence_duration * 2):
                        should_transcribe = True
                        
                # Condition 2: The window is "full"
                if window_len == self.SLIDING_WINDOW_SAMPLES:
                    should_transcribe = True
                
                # If condition triggered => transcribe
                if should_transcribe:
                    return await self._sliding_window_transcribe(finalize=not self.is_speaking)
            
            # If we are not transcribing yet, return partial “buffering” info
            return {
                "type": "buffering",
                "text": self.last_partial_text,
                "accumulated_text": self.accumulated_text,
                "is_speaking": self.is_speaking
            }
        
        except Exception as e:
            logger.error(f"process_chunk error: {e}")
            raise
    
    async def _sliding_window_transcribe(self, finalize: bool = False) -> Dict[str, Any]:
        """
        Transcribes the last ~3s from `sliding_window_buffer`,
        compares to old partial, merges if better, 
        optionally finalizes if user has stopped speaking.
        """
        # Copy the current buffer for transcription
        audio_to_transcribe = np.copy(self.sliding_window_buffer)
        
        # In practice, you might want to do overlap with more 
        # than 3s or crossfade; for now, we do a direct pass.
        result = await self._run_whisper_transcribe(audio_to_transcribe)
        
        # Compare confidence
        current_confidence = self._calculate_confidence(result)
        logger.info(f"Sliding window transcription confidence: {current_confidence:.2f}")
        
        # If new is better, replace partial text. If not, keep old partial.
        if current_confidence > self.last_partial_confidence:
            self.last_partial_text = result.get("text", "").strip()
            self.last_partial_confidence = current_confidence
        
        # If finalizing (user silence), we merge the partial text into the accumulated text
        if finalize:
            # Merge last partial to final text
            self._merge_into_accumulated(self.last_partial_text)
            
            # Reset partial states
            final_result = {
                "type": "final",
                "text": self.accumulated_text,
                "segments": [],  # you can fill this with more detail
                "confidence": self.last_partial_confidence
            }
            
            self._reset_utterance_state()
            logger.info("Finalized transcription after silence.")
            return final_result
        else:
            # Return partial result but do not finalize
            partial_result = {
                "type": "partial",
                "partial_text": self.last_partial_text,
                "accumulated_text": self.accumulated_text,
                "confidence": self.last_partial_confidence
            }
            return partial_result
    
    async def _finalize_transcription(self) -> Dict[str, Any]:
        """
        Called when the stream is ended or forcibly closed:
        - We'll do one last transcription if needed
        - Then finalize everything
        """
        # If we have enough samples in the buffer to transcribe, do it
        if len(self.sliding_window_buffer) >= self.MIN_TRANSCIBE_WINDOW_SAMPLES:
            result = await self._run_whisper_transcribe(self.sliding_window_buffer)
            final_conf = self._calculate_confidence(result)
            if final_conf > self.last_partial_confidence:
                self.last_partial_text = result.get("text", "").strip()
                self.last_partial_confidence = final_conf
        
        # Merge partial into final
        self._merge_into_accumulated(self.last_partial_text)
        
        final_result = {
            "type": "final",
            "text": self.accumulated_text,
            "segments": [],
            "confidence": self.last_partial_confidence
        }
        self._reset_utterance_state()
        return final_result
    
    def _merge_into_accumulated(self, new_text: str):
        """
        Minimal text overlap check to avoid repeating partial words.
        A more advanced approach might do partial alignment.
        """
        if not new_text:
            return
        
        # Example logic: if the new text is mostly repeated in
        # the last 10 words of accumulated_text, skip or partial-merge.
        last_10_words = " ".join(self.accumulated_text.lower().split()[-10:])
        new_words = new_text.lower().split()
        
        overlap = sum(1 for w in new_words if w in last_10_words)
        # If overlap is too big, we skip. Otherwise, we append.
        if len(new_words) > 0 and overlap / len(new_words) > 0.7:
            logger.debug("Detected large overlap; minimal text appended.")
        else:
            self.accumulated_text = (self.accumulated_text + " " + new_text).strip()
    
    def _reset_utterance_state(self):
        """
        Reset states after finishing an utterance, so we can start fresh.
        """
        self.sliding_window_buffer = np.array([], dtype=np.float32)
        self.last_partial_text = ""
        self.last_partial_confidence = 0.0
        self.currently_in_utterance = False
        self.is_speaking = False
    
    async def end_session(self) -> Dict[str, Any]:
        """
        If the websocket or stream ends abruptly, finalize transcription.
        """
        return await self._finalize_transcription()
    
    # -------------------------------------------
    # HELPER METHODS
    # -------------------------------------------
    async def _run_whisper_transcribe(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Offload the synchronous transcribe call to a thread executor
        to avoid blocking the event loop.
        """
        loop = torch._C._get_tracing_state() or None
        # If using normal python, do:
        # loop = asyncio.get_event_loop() 
        return await (loop or asyncio.get_event_loop()).run_in_executor(
            None,
            self.processor.transcribe_sync,
            audio_data
        )
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Compute an approximate confidence from the Whisper result.
        You can use your existing method or more advanced logic here.
        """
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # Example: average of exponential(logprob)
        conf_values = []
        for seg in segments:
            avg_lp = seg.get("avg_logprob", -2.0)
            no_speech = seg.get("no_speech_prob", 0.0)
            # Convert log prob to [0..1], penalize no_speech
            seg_conf = np.exp(avg_lp) * (1.0 - no_speech)
            conf_values.append(seg_conf)
        
        if not conf_values:
            return 0.0
        
        return float(np.mean(conf_values))


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