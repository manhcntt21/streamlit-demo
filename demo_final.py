import streamlit as st
import cv2
import numpy as np
import os
import threading
import time
import sounddevice as sd
from retinaface import RetinaFace
import pyaudio
import wave
from openai import OpenAI
import io
import numpy as np

from speech.stt_stream import transcribe_audio


# Your OpenAI API key
api_key = ''

# Microphone settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (Hz)
CHUNK = 16000  # Size of each audio chunk
DEVICE_INDEX = None  # You can set this if you have a specific microphone device

# Create a PyAudio object
p = pyaudio.PyAudio()

client = OpenAI(api_key=api_key)

# Function to read audio file in chunks
wav_file_path = '/home/tony/Downloads/qh (mp3cut.net).wav'


# set page config
st.set_page_config(page_title="Chatbot TLU", page_icon="🦈", layout="wide")

# Stream video from webcam
st.title("Webcam Live Feed")

# Khởi tạo các biến session state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "frame_window" not in st.session_state:
    st.session_state.frame_window = st.empty()

# Cấu hình audio
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100


class AudioVideoRecorder:
    def __init__(self):
        self.video_capture = None
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.check_face_detection = True

    def start_recording(self):
        # Khởi tạo video capture
        self.video_capture = cv2.VideoCapture(0)

        # Khởi tạo audio stream
        self.audio_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        self.is_recording = True

        # Bắt đầu thread ghi âm
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def record_audio(self):
        while self.is_recording:
            try:
                data = self.audio_stream.read(CHUNK)
                yield data
                # self.frames.append(data)
            except Exception as e:
                st.error(f"Lỗi ghi âm: {str(e)}")
                break

    def get_video_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()

            if ret:
                if self.check_face_detection:
                    # faces = RetinaFace.detect_faces(frame)
                    faces = True
                    if faces:
                        print("Face detected")
                        # for face in faces.values():
                        #     x1, y1, x2, y2 = face['facial_area']
                        #     # Draw a rectangle around the face
                        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        say_hello()
                        self.check_face_detection = False

                        voice_str, count = '', 0
                        # ====Function to stream audio and transcribe it========
                        for audio_chunk in self.record_audio():
                            text = transcribe_audio(audio_chunk)
                            if text:
                                print("You said:", text)
                            else:
                                print("Sorry, could not understand the audio.")

                            if text != "Hẹn gặp lại các bạn trong những video tiếp theo nhé!":
                                voice_str += text + ' '
                                count = 0
                            else:
                                count += 1

                            if count >= 5:
                                print("Text processing")
                                print("Voice speech")
                                count = 0
                                voice_str = ''

                    else:
                        print("No face detected")

                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def stop_recording(self):
        self.is_recording = False

        # Dừng và giải phóng video capture
        if self.video_capture is not None:
            self.video_capture.release()

        # Dừng và giải phóng audio stream
        if hasattr(self, "audio_stream"):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        # create a folder if not exist
        if not os.path.exists("../demo_final/recorded_audio"):
            os.makedirs("../demo_final/recorded_audio")

        # # # Lưu file audio
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"rrecorded_audio/ecording_{timestamp}.wav"
        #
        # wf = wave.open(filename, "wb")
        # wf.setnchannels(CHANNELS)
        # wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        # wf.setframerate(RATE)
        # wf.writeframes(b"".join(self.frames))
        # wf.close()

        return filename


def say_hello():
    print("Hello World!")
    return "hello"


def audio_to_text(filename):
    return "text"


def text_to_audio():
    pass


def load_wav_file(wav_file):
    try:
        wav_bytes = io.BytesIO(wav_file.read())

        with wave.open(wav_bytes, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()

        # read raw audio
        raw_data = wav.readframes(n_frames)

        # convert raw data to numpy array
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        else:
            raise ValueError("Unsupported sample width")

        audio_data = np.frombuffer(raw_data, dtype=dtype)

        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)

        return audio_data, framerate, channels
    except Exception as e:
        st.error(f"Error loading WAV file: {str(e)}")
        return None, None, None


def play_audio(audio_data, framerate):
    """
    play audio using sounddevice
    """
    try:
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        sd.play(audio_data, framerate)
        sd.wait()
    except Exception as e:
        st.error(f"Error playing audio")


if __name__ == "__main__":
    # sidebar
    with st.sidebar:
        st.title("🦙💬 TLU Chatbot")
        st.write(
            "This chatbot is created using the open-source LLM model and built from Ollama. "
        )

        # main page
    st.markdown("# Welcome to the Chatbot TLU! :sunglasses: :sunglasses:")

    # Khởi tạo recorder nếu chưa có
    if "recorder" not in st.session_state:
        st.session_state.recorder = AudioVideoRecorder()

    # Nút điều khiển ghi hình
    if st.button("Toggle Recording"):
        if not st.session_state.recording:
            st.session_state.recording = True
            st.session_state.recorder = AudioVideoRecorder()
            st.session_state.recorder.start_recording()

        else:
            st.session_state.recording = False
            filename = st.session_state.recorder.stop_recording()
            st.success(f"Đã lưu recording vào file: {filename}")
            # # audio_to_text(filename)
            # # text_to_audio()
            # # print(filename)
            # # audio_data, framerate, channels = load_wav_file('recording_20241106_142205.wav')
            # # play_audio(audio_data, framerate)
            st.audio(filename)

    # Hiển thị trạng thái
    status_text = "Đang ghi" if st.session_state.recording else "Đã dừng"
    st.text(f"Trạng thái: {status_text}")

    # Hiển thị video frame
    while st.session_state.recording:
        frame = st.session_state.recorder.get_video_frame()
        if frame is not None:
            st.session_state.frame_window.image(frame)
            time.sleep(1 / 60)  # Giới hạn ~30 FPS
            # st.rerun()
    st.session_state.frame_window.write("Camera đã tắt")
