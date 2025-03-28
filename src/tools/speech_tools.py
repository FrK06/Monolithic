# src/tools/speech_tools.py

import os
import tempfile
import base64
from typing import Tuple, Optional
from pathlib import Path
import logging

# Import OpenAI for Whisper API (STT) and TTS
from openai import OpenAI

logger = logging.getLogger(__name__)

class SpeechTools:
    def __init__(self):
        """Initialize speech tools with OpenAI client."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.voice = "alloy"  # Default voice - options: alloy, echo, fable, onyx, nova, shimmer
        
    def speech_to_text(self, audio_data: bytes) -> Tuple[str, Optional[str]]:
        """
        Convert speech to text using OpenAI's Whisper model.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Tuple with transcribed text and any error message
        """
        try:
            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_path = temp_audio.name
            
            # Use OpenAI Whisper API for transcription
            with open(temp_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Clean up the temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            # Return the transcript
            return transcript.text, None
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {str(e)}")
            return "", f"Speech recognition error: {str(e)}"
    
    def text_to_speech(self, text: str) -> Tuple[bytes, Optional[str]]:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: Text content to convert to speech
            
        Returns:
            Tuple with audio data in bytes and any error message
        """
        try:
            # Generate speech using OpenAI TTS API
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text
            )
            
            # Get the audio content
            audio_data = response.content
            
            return audio_data, None
            
        except Exception as e:
            logger.error(f"Error in text_to_speech: {str(e)}")
            return b"", f"Text-to-speech error: {str(e)}"
    
    def set_voice(self, voice: str) -> bool:
        """
        Set the TTS voice.
        
        Args:
            voice: Voice identifier (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            Boolean indicating success
        """
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice in valid_voices:
            self.voice = voice
            return True
        return False
    
    def decode_base64_audio(self, base64_audio: str) -> bytes:
        """
        Decode base64 audio data.
        
        Args:
            base64_audio: Base64 encoded audio string
            
        Returns:
            Decoded audio bytes
        """
        try:
            # Remove potential prefix
            if ',' in base64_audio:
                base64_audio = base64_audio.split(',')[1]
            
            # Decode base64 string to bytes
            return base64.b64decode(base64_audio)
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {str(e)}")
            raise ValueError(f"Failed to decode audio data: {str(e)}")
    
    def encode_audio_to_base64(self, audio_data: bytes) -> str:
        """
        Encode audio data to base64 string for frontend.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Base64 encoded audio string with appropriate data URI prefix
        """
        try:
            b64_encoded = base64.b64encode(audio_data).decode('utf-8')
            return f"data:audio/mp3;base64,{b64_encoded}"
        except Exception as e:
            logger.error(f"Error encoding audio to base64: {str(e)}")
            raise ValueError(f"Failed to encode audio data: {str(e)}")
