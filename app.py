import gradio as gr
import google.generativeai as genai
from gtts import gTTS
import os
import numpy as np
import tempfile
import speech_recognition as sr
from scipy.io import wavfile
import io
import pyaudio
import wave
import time
from pydantic import BaseModel
from typing import Optional, List
from starlette.requests import Request

class Message(BaseModel):
    role: str
    content: str

    class Config:
        arbitrary_types_allowed = True

class VoiceAssistantConfig(BaseModel):
    request: Optional[Request] = None
    api_key: Optional[str] = None
    messages: List[Message] = []

    class Config:
        arbitrary_types_allowed = True

class VoiceAssistant:
    def __init__(self):
        self.model = None
        self.config = VoiceAssistantConfig()
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 5

    def configure_gemini(self, api_key: str) -> str:
        """Configure Gemini AI with the provided API key"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.config.api_key = api_key
            return "API key configured successfully!"
        except Exception as e:
            return f"Error configuring API: {str(e)}"

    def record_audio(self) -> str:
        """Record audio using PyAudio"""
        p = pyaudio.PyAudio()

        stream = p.open(format=self.format,
                       channels=self.channels,
                       rate=self.rate,
                       input=True,
                       frames_per_buffer=self.chunk)

        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            wf = wave.open(temp_audio.name, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            return temp_audio.name

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Google Speech Recognition"""
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            return f"Error in transcription: {str(e)}"

    def text_to_speech(self, text: str) -> str:
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                tts.write_to_fp(temp_audio)
                return temp_audio.name
        except Exception as e:
            return f"Error in text-to-speech conversion: {str(e)}"

    def process_audio(self, audio_path: Optional[str], api_key: str) -> tuple:
        """Process audio input and generate response"""
        if not self.model:
            result = self.configure_gemini(api_key)
            if "Error" in result:
                return None, None, result

        try:
            # If audio_path is None (record new), else use uploaded file
            if audio_path is None:
                audio_path = self.record_audio()

            # Transcribe audio
            transcribed_text = self.transcribe_audio(audio_path)
            if isinstance(transcribed_text, str) and transcribed_text.startswith("Error"):
                return None, None, transcribed_text

            # Get response from Gemini
            response = self.model.generate_content(transcribed_text)
            response_text = response.text

            # Convert response to speech
            response_audio = self.text_to_speech(response_text)

            # Update chat history with proper Message models
            self.config.messages.append(Message(role="user", content=transcribed_text))
            self.config.messages.append(Message(role="assistant", content=response_text))

            # Format chat history
            chat_history = "\n".join([
                f"{'You' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in self.config.messages
            ])

            return response_audio, chat_history, None

        except Exception as e:
            return None, None, f"Error generating response: {str(e)}"
        finally:
            # Cleanup temporary files
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

    def clear_history(self) -> tuple:
        """Clear chat history"""
        self.config.messages = []
        return None, "Chat history cleared!"

def create_voice_assistant_interface() -> gr.Blocks:
    """Create Gradio interface for voice assistant"""
    assistant = VoiceAssistant()

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# LumiGem: 🎤 Voice AI Assistant")

        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="Enter your API key here..."
                )

                with gr.Row():
                    record_btn = gr.Button("🎤 Record (5s)")
                    upload_btn = gr.File(label="Upload Audio")

                clear_btn = gr.Button("Clear Chat History")

            with gr.Column(scale=2):
                chat_output = gr.Textbox(
                    label="Chat History",
                    lines=10,
                    interactive=False
                )

                audio_output = gr.Audio(
                    label="AI Response",
                    type="filepath"
                )

                error_output = gr.Textbox(
                    label="Status/Error Messages",
                    visible=True
                )

        # Handle record button
        record_btn.click(
            fn=lambda x: assistant.process_audio(None, x),
            inputs=[api_key_input],
            outputs=[audio_output, chat_output, error_output]
        )

        # Handle file upload
        upload_btn.upload(
            fn=assistant.process_audio,
            inputs=[upload_btn, api_key_input],
            outputs=[audio_output, chat_output, error_output]
        )

        # Handle clear history
        clear_btn.click(
            fn=assistant.clear_history,
            inputs=[],
            outputs=[audio_output, chat_output]
        )

        # Configure API key
        api_key_input.change(
            fn=assistant.configure_gemini,
            inputs=[api_key_input],
            outputs=[error_output]
        )

    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_voice_assistant_interface()
    demo.launch(share=True, debug=True)
