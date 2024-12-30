import numpy as np
from faster_whisper import WhisperModel
import torch
import torch.nn as nn
from typing import Generator, List, Optional, Union, Callable
import wave
import io
import threading
from queue import Queue
import time

class AudioConfig:
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 sample_width: int = 2,
                 chunk_duration: float = 30.0,
                 vad_threshold: float = 0.5,
                 silence_duration: float = 1.0,
                 realtime_updates: bool = False,
                 update_interval: float = 0.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.chunk_duration = 5.0 if realtime_updates else chunk_duration
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.realtime_updates = realtime_updates
        self.update_interval = update_interval

def get_available_device():
    """Determine the best available device for PyTorch."""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return "cuda"
        except Exception as e:
            print(f"CUDA is available but encountered an error: {e}")
            print("Falling back to CPU")
    return "cpu"

class FasterWhisperStreamingClient:
    def __init__(self, 
                 model_name: str = "base",
                 audio_config: Optional[AudioConfig] = None,
                 device: Optional[str] = None,
                 compute_type: str = "float32",
                 cpu_threads: int = 4,
                 on_recording_start: Optional[Callable] = None,
                 on_recording_stop: Optional[Callable] = None,
                 on_realtime_transcription_update: Optional[Callable] = None):
        
        self.device = device if device else get_available_device()
        if self.device == "cpu":
            compute_type = "float32"
        
        try:
            self.model = WhisperModel(
                model_size_or_path=model_name,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=cpu_threads
            )
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to CPU with float32")
            self.device = "cpu"
            compute_type = "float32"
            self.model = WhisperModel(
                model_size_or_path=model_name,
                device="cpu",
                compute_type="float32",
                cpu_threads=cpu_threads
            )

        self.audio_config = audio_config or AudioConfig()
        
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True,
                onnx=False
            )
            self.vad_model.to(self.device)
        except Exception as e:
            print(f"Error loading VAD model on {self.device}: {e}")
            print("Falling back to CPU for VAD")
            self.device = "cpu"
            self.vad_model = self.vad_model.to("cpu")
        
        self.get_speech_timestamps = utils[0]
        
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_transcription_update = on_realtime_transcription_update
        
        self._queue = Queue()
        self._stop_flag = threading.Event()
        self._processing_thread = None
        self._is_speaking = False
        self._last_voice_timestamp = 0
        self._voice_buffer = np.array([], dtype=np.float32)
        self._last_update_time = 0
        self._current_segment = ""
        
        print(f"Initialized with device: {self.device}, compute type: {compute_type}")
        if self.audio_config.realtime_updates:
            print(f"Real-time updates enabled with {self.audio_config.update_interval}s interval")

    def _check_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        try:
            audio_tensor = torch.from_numpy(audio_chunk).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_tensor = audio_tensor.to(self.device)
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=self.audio_config.vad_threshold,
                sampling_rate=self.audio_config.sample_rate
            )
            
            has_speech = len(speech_timestamps) > 0
            current_time = time.time()
            
            if has_speech:
                if not self._is_speaking:
                    self._is_speaking = True
                    if self.on_recording_start:
                        self.on_recording_start()
                self._last_voice_timestamp = current_time
            elif self._is_speaking and (current_time - self._last_voice_timestamp > self.audio_config.silence_duration):
                self._is_speaking = False
                if self.on_recording_stop:
                    self.on_recording_stop()
            
            return has_speech
        except Exception as e:
            print(f"Error in voice activity detection: {e}")
            return False

    def _should_update(self) -> bool:
        if not self.audio_config.realtime_updates:
            return False
            
        current_time = time.time()
        if current_time - self._last_update_time >= self.audio_config.update_interval:
            self._last_update_time = current_time
            return True
        return False

    def start(self, audio_generator: Generator[bytes, None, None]) -> Generator[dict, None, None]:
        """Start processing audio stream."""
        self._stop_flag.clear()
        self._processing_thread = threading.Thread(
            target=self._process_audio_stream,
            args=(audio_generator,)
        )
        self._processing_thread.start()
        
        try:
            while not self._stop_flag.is_set() or not self._queue.empty():
                try:
                    result = self._queue.get(timeout=1.0)
                    if self.on_transcription_update:
                        self.on_transcription_update(result)
                    yield result
                except Queue.Empty:
                    # Check if processing is complete
                    if not self._processing_thread.is_alive():
                        break
                    continue
        finally:
            self.end()

    def _process_audio_stream(self, audio_generator: Generator[bytes, None, None]):
        """Process incoming audio stream with VAD and transcription."""
        chunk_samples = int(self.audio_config.sample_rate * self.audio_config.chunk_duration)
        update_samples = int(self.audio_config.sample_rate * self.audio_config.update_interval)
        
        try:
            for chunk in audio_generator:
                if self._stop_flag.is_set():
                    break
                    
                try:
                    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    has_speech = self._check_voice_activity(audio_data)
                    
                    if has_speech:
                        self._voice_buffer = np.concatenate([self._voice_buffer, audio_data])
                        
                        if self.audio_config.realtime_updates and self._should_update() and len(self._voice_buffer) >= update_samples:
                            result = self._transcribe_chunk(self._voice_buffer, is_final=False)
                            self._queue.put(result)
                        
                        if len(self._voice_buffer) >= chunk_samples:
                            result = self._transcribe_chunk(self._voice_buffer, is_final=True)
                            self._queue.put(result)
                            self._voice_buffer = np.array([], dtype=np.float32)
                            
                    elif len(self._voice_buffer) > 0:
                        result = self._transcribe_chunk(self._voice_buffer, is_final=True)
                        self._queue.put(result)
                        self._voice_buffer = np.array([], dtype=np.float32)
                        
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
                    continue
            
            # Process any remaining audio in the buffer
            if len(self._voice_buffer) > 0:
                try:
                    result = self._transcribe_chunk(self._voice_buffer, is_final=True)
                    self._queue.put(result)
                except Exception as e:
                    print(f"Error processing final chunk: {e}")
        
        finally:
            # Signal that processing is complete
            self._stop_flag.set()

    def _transcribe_chunk(self, audio_chunk: np.ndarray, is_final: bool = True) -> dict:
        try:
            segments, info = self.model.transcribe(
                audio_chunk,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True
            )
            
            text_segments = []
            for segment in segments:
                text_segments.append({
                    'text': segment.text,
                    'start': segment.start,
                    'end': segment.end,
                    'words': [{'word': word.word, 'start': word.start, 'end': word.end} 
                             for word in segment.words]
                })
            
            if text_segments:
                self._current_segment = text_segments[-1]['text']
            
            return {
                'type': 'partial' if not is_final else 'final',
                'segments': text_segments,
                'language': info.language,
                'timestamp': time.time(),
                'is_speaking': self._is_speaking,
                'current_segment': self._current_segment
            }
        except Exception as e:
            print(f"Error in transcription: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'timestamp': time.time(),
                'is_speaking': self._is_speaking
            }

    def end(self):
        """Stop the streaming process and cleanup."""
        self._stop_flag.set()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)  # Wait up to 5 seconds
            
        # Clear any remaining items in the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Queue.Empty:
                break

def main():
    def on_recording_start():
        print("\nVoice activity detected - Recording started")
        
    def on_recording_stop():
        print("\nVoice activity ended - Recording stopped")
        
    def on_transcription_update(result):
        if result.get('type') == 'error':
            print(f"\nError: {result['error']}")
        else:
            if result['type'] == 'partial':
                print(f"\r[PARTIAL] {result['current_segment']}", end='', flush=True)
            else:
                print(f"\n[FINAL] {result['current_segment']}")
    
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        realtime_updates=False,  # Set to False for regular mode
        update_interval=0.5
    )
    
    client = FasterWhisperStreamingClient(
        model_name="base",
        audio_config=config,
        compute_type="float32",
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_realtime_transcription_update=on_transcription_update
    )
    
    def audio_generator(file_path):
        chunk_size = 1600 if config.realtime_updates else 32000
        with open(file_path, 'rb') as audio_file:
            # while chunk := audio_file.read(chunk_size):
            for chunk in iter(lambda: audio_file.read(chunk_size), b''):
                yield chunk
    
    # Start streaming
    for result in client.start(audio_generator("prompt.wav")):
        pass  # Results are handled by callbacks
    
    client.end()

if __name__ == "__main__":
    main()