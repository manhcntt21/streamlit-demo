import streamlit as st
import cv2
import numpy as np
import pyaudio
from datetime import datetime
import wave
import threading
import sounddevice as sd
import io
import queue
from contextlib import contextmanager

# Set page config
st.set_page_config(page_title='Chatbot TLU', page_icon="🦈", layout='wide')

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

class AudioVideoRecorder:
    def __init__(self):
        self.video_capture = None
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stop_flag = threading.Event()

    def start_recording(self):
        try:
            # Initialize video capture
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise ValueError("Could not open webcam")

            # Initialize audio stream
            self.audio_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            self.is_recording = True
            self.stop_flag.clear()

            # Start audio recording thread
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.start()
            
            return True
        except Exception as e:
            st.error(f"Error starting recording: {str(e)}")
            self.cleanup()
            return False

    def _record_audio(self):
        while not self.stop_flag.is_set():
            try:
                data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                if not self.stop_flag.is_set():
                    st.error(f"Audio recording error: {str(e)}")
                break

    def get_video_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'video_capture') and self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass

        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except Exception:
                pass

    def stop_recording(self):
        """Stop recording and save file"""
        if not self.is_recording:
            return None

        self.stop_flag.set()
        self.is_recording = False

        # Wait for audio thread to complete
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)

        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
        except Exception as e:
            st.error(f"Error saving audio: {str(e)}")
            filename = None

        # Clean up resources
        self.cleanup()
        return filename

@contextmanager
def audio_player_container():
    """Container for managing audio player state"""
    container = st.empty()
    try:
        yield container
    finally:
        container.empty()

def main():
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioVideoRecorder()

    with audio_player_container() as container:
        st.session_state.audio_container = container
        container.audio("filename.wav")

    # Display status
    status_text = "Recording" if st.session_state.recording else "Stopped"
    st.text(f"Status: {status_text}")

    # Display video frame
    if st.session_state.recording and st.session_state.recorder:
        frame = st.session_state.recorder.get_video_frame()
        if frame is not None:
            st.session_state.frame_window.image(frame)
        st.rerun()
    else:
        st.session_state.frame_window.empty()

if __name__ == '__main__':
    main()