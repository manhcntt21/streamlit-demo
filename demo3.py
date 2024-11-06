import streamlit as st
import cv2
import numpy as np
import time
import pyaudio
import wave
import threading
import os
from datetime import datetime

# Set page config first
st.set_page_config(page_title="Chatbot TLU", page_icon="🦈", layout="wide")

# Initialize session state variables using if statements
if 'audio_recording' not in st.session_state:
    st.session_state.audio_recording = False
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'audio_thread' not in st.session_state:
    st.session_state.audio_thread = None
if 'recorded_files' not in st.session_state:
    st.session_state.recorded_files = []
if 'recording_in_progress' not in st.session_state:
    st.session_state.recording_in_progress = False
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio
@st.cache_resource
def get_pyaudio():
    return pyaudio.PyAudio()

audio = get_pyaudio()

# Create a directory for saving audio files if it doesn't exist
if not os.path.exists("recorded_audio"):
    os.makedirs("recorded_audio")

# sidebar
with st.sidebar:
    st.title("🦙💬 TLU Chatbot")
    st.write(
        "This chatbot is created using the open-source LLM model and built from Ollama. "
    )

# main page
st.markdown("# Welcome to the Chatbot TLU! :sunglasses: :sunglasses:")

def record_audio():
    """Function to record audio"""
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
    
    frames = []
    
    while st.session_state.audio_recording:
        try:
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Calculate and display recording duration
            duration = time.time() - st.session_state.recording_start_time
            st.write(f"Recording duration: {duration:.2f} seconds")
            
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
            break
    
    stream.stop_stream()
    stream.close()
    
    if frames:  # Only save if we have recorded frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_audio/audio_{timestamp}.wav"
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        st.session_state.recorded_files.append(filename)
        st.session_state.recording_in_progress = False
        return filename
    return None

# Create columns for camera and audio controls
col1, col2 = st.columns(2)

# Camera controls
with col1:
    st.subheader("Camera Controls")
    
    if not st.session_state.camera_running:
        camera_available = True
        try:
            # Check if camera is available
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                camera_available = False
            cap.release()
        except:
            camera_available = False
        
        if camera_available:
            start_cam = st.button("Start Camera", key='start_camera')
            if start_cam:
                st.session_state.camera_running = True
        else:
            st.warning("Camera not available")
    else:
        stop_cam = st.button("Stop Camera", key='stop_camera')
        if stop_cam:
            st.session_state.camera_running = False

# Audio controls
with col2:
    st.subheader("Audio Controls")
    if not st.session_state.recording_in_progress:
        start_rec = st.button("Start Recording", key='start_recording')
        if start_rec:
            st.session_state.audio_recording = True
            st.session_state.recording_in_progress = True
            st.session_state.recording_start_time = time.time()
            st.session_state.audio_thread = threading.Thread(target=record_audio)
            st.session_state.audio_thread.start()
    else:
        stop_rec = st.button("Stop Recording", key='stop_recording')
        if stop_rec:
            st.session_state.audio_recording = False
            if st.session_state.audio_thread:
                st.session_state.audio_thread.join()
                st.success("Audio recording saved!")

# Display camera feed
FRAME_WINDOW = st.empty()

# Camera handling
if st.session_state.camera_running:
    camera = cv2.VideoCapture(0)

    try:
        while st.session_state.camera_running:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from camera")
                st.session_state.camera_running = False
                break

            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

    except Exception as e:
        st.error(f"Error accessing camera: {str(e)}")

    finally:
        camera.release()
else:
    st.write("Camera Stopped")

# Display recording status
if st.session_state.recording_in_progress:
    st.write("🎤 Recording in progress...")
else:
    st.write("Audio Recording Stopped")

# Display recorded files
if st.session_state.recorded_files:
    st.subheader("Recorded Audio Files")
    for idx, file in enumerate(st.session_state.recorded_files):
        if os.path.exists(file):  # Check if file exists
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                with open(file, 'rb') as f:
                    st.audio(f.read(), format='audio/wav')
            with col2:
                # Add unique key for delete button
                if st.button(f"Delete", key=f"delete_{idx}"):
                    os.remove(file)
                    st.session_state.recorded_files.remove(file)
            with col3:
                # Add download button
                with open(file, 'rb') as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=os.path.basename(file),
                        mime='audio/wav',
                        key=f"download_{idx}"
                    )

# text input
st.markdown("Type your message below:")