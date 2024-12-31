# test_whisper_server.py
import asyncio
import websockets
import aiohttp
import json
import wave
import time
import argparse
import logging
from typing import Generator
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperServerTester:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def create_audio_chunks(self, audio_file: str, chunk_size: int = 4000) -> Generator[bytes, None, None]:
        """Generate audio chunks from a WAV file."""
        with wave.open(audio_file, 'rb') as wav:
            while True:
                chunk = wav.readframes(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def test_async_api_url(self, audio_url: str) -> bool:
        """Test the async API endpoint using a URL."""
        try:
            logger.info("Creating transcription job using URL...")
            async with self.session.post(
                f"{self.base_url}/v1/jobs",
                json={
                    "source_config": {"url": audio_url},
                    "metadata": "Test transcription via URL"
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to create job: {await response.text()}")
                    return False
                
                job_data = await response.json()
                return await self._poll_job_status(job_data["id"])

        except Exception as e:
            logger.error(f"Error testing async API with URL: {str(e)}")
            return False

    async def test_async_api_file(self, audio_file: str) -> bool:
        """Test the async API endpoint using file upload."""
        try:
            logger.info("Creating transcription job using file upload...")
            
            # Prepare multipart form data
            form = aiohttp.FormData()
            form.add_field('file',
                         open(audio_file, 'rb'),
                         filename=os.path.basename(audio_file),
                         content_type='audio/wav')
            form.add_field('metadata', 'Test transcription via file upload')
            
            async with self.session.post(
                f"{self.base_url}/v1/jobs/upload",
                data=form
            ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to create job: {await response.text()}")
                        return False
                    
                    job_data = await response.json()
                    return await self._poll_job_status(job_data["id"])

        except Exception as e:
            logger.error(f"Error testing async API with file upload: {str(e)}")
            return False

    async def _poll_job_status(self, job_id: str) -> bool:
        """Poll for job completion."""
        logger.info(f"Job created with ID: {job_id}")
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            async with self.session.get(f"{self.base_url}/v1/jobs/{job_id}") as response:
                if response.status != 200:
                    logger.error(f"Failed to get job status: {await response.text()}")
                    return False
                
                status_data = await response.json()
                status = status_data["status"]
                
                if status == "completed":
                    logger.info("Job completed successfully!")
                    logger.info(f"Transcription result: {status_data.get('result', {}).get('text', '')}")
                    return True
                elif status == "error":
                    logger.error(f"Job failed: {status_data.get('error')}")
                    return False
                
                logger.info(f"Job status: {status}")
                await asyncio.sleep(2)
                attempt += 1

        logger.error("Job timed out")
        return False

    async def test_streaming_api(self, audio_file: str) -> bool:
        """Test the streaming API endpoint."""
        websocket = None
        try:
            logger.info("Testing streaming API...")
            websocket = await websockets.connect(f"{self.ws_url}/v1/stream")
            
            # Send audio chunks
            for chunk in self.create_audio_chunks(audio_file):
                try:
                    await websocket.send(chunk)
                    
                    # Receive and process results
                    try:
                        result = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        result_data = json.loads(result)
                        
                        if result_data.get("type") == "error":
                            logger.error(f"Streaming error: {result_data.get('error')}")
                        elif result_data.get("type") == "final":
                            logger.info(f"Transcription: {result_data.get('text', '')}")
                        
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed by server")
                        break
                    except Exception as e:
                        logger.error(f"Error processing result: {str(e)}")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    break

            # Send empty chunk to signal end of stream
            try:
                await websocket.send(b"")
                
                # Wait for final results
                try:
                    while True:
                        result = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        result_data = json.loads(result)
                        if result_data.get("type") == "final":
                            logger.info(f"Final transcription: {result_data.get('text', '')}")
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass
                    
            except websockets.exceptions.ConnectionClosed:
                pass

            logger.info("Streaming test completed")
            return True

        except Exception as e:
            logger.error(f"Error testing streaming API: {str(e)}")
            return False
            
        finally:
            if websocket:
                try:
                    await websocket.close()
                except:
                    pass

async def run_tests(args):
    """Run all tests."""
    async with WhisperServerTester(args.host, args.port) as tester:
        # Test async API
        if args.test_async:
            logger.info("\n=== Testing Async API ===")
            if args.audio_url:
                logger.info("Testing with URL...")
                success = await tester.test_async_api_url(args.audio_url)
                logger.info(f"Async API (URL) test {'succeeded' if success else 'failed'}")
            
            if args.audio_file:
                logger.info("Testing with file upload...")
                success = await tester.test_async_api_file(args.audio_file)
                logger.info(f"Async API (file) test {'succeeded' if success else 'failed'}")

        # Test streaming API
        if args.test_streaming and args.audio_file:
            logger.info("\n=== Testing Streaming API ===")
            success = await tester.test_streaming_api(args.audio_file)
            logger.info(f"Streaming API test {'succeeded' if success else 'failed'}")

def main():
    parser = argparse.ArgumentParser(description="Test Whisper STT Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--audio-url", help="URL of audio file for async API testing")
    parser.add_argument("--audio-file", help="Path to local audio file for testing")
    parser.add_argument("--test-async", action="store_true", help="Test async API")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming API")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_async and not (args.audio_url or args.audio_file):
        parser.error("Either --audio-url or --audio-file is required for async API testing")
    if args.test_streaming and not args.audio_file:
        parser.error("--audio-file is required for streaming API testing")
    
    asyncio.run(run_tests(args))

if __name__ == "__main__":
    main()