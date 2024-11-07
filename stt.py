import whisper
import numpy as np
import wave
import time
from pydub import AudioSegment
from io import BytesIO
import tempfile

# Load Whisper model
model = whisper.load_model("base")  # You can use "small", "medium", or "large" models

# Audio file path
audio_file = "/home/tony/Downloads/qh.wav"

# Parameters for chunk processing
SAMPLE_RATE = 16000  # 16 kHz (recommended by Whisper)
LANGUAGE = "vi"  # Specify the language as Vietnamese (ISO 639-1 code for Vietnamese)


def process_audio_chunk(chunk):
    # Write chunk to a temporary in-memory file
    audio_np = np.frombuffer(chunk, dtype=np.int16)

    # Create a temporary file to write WAV data in memory
    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp_wav_file:
        # Write the raw audio data to the temp file
        wav = wave.open(tmp_wav_file, 'wb')
        wav.setnchannels(1)  # Mono audio
        wav.setsampwidth(2)  # 16-bit samples
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_np.tobytes())

        # Move the pointer back to the beginning of the file
        tmp_wav_file.seek(0)

        # Now, transcribe the temporary WAV file using Whisper
        result = model.transcribe(tmp_wav_file.name, language=LANGUAGE)

    # Return the transcribed text
    return result['text']


def stream_audio_from_file(file_path):
    # Load the audio file using pydub
    audio = AudioSegment.from_wav(file_path)

    # Resample if necessary (ensure the sample rate is 16kHz)
    if audio.frame_rate != SAMPLE_RATE:
        print(f"Resampling from {audio.frame_rate}Hz to {SAMPLE_RATE}Hz")
        audio = audio.set_frame_rate(SAMPLE_RATE)

    # Split audio into chunks (e.g., 1 second per chunk)
    chunk_duration_ms = 1000  # 1 second per chunk
    total_duration_ms = len(audio)
    num_chunks = total_duration_ms // chunk_duration_ms

    for i in range(num_chunks):
        chunk = audio[i * chunk_duration_ms:(i + 1) * chunk_duration_ms]

        # Export chunk to raw audio bytes
        chunk_bytes = chunk.raw_data

        # Process the chunk for transcription
        transcription = process_audio_chunk(chunk_bytes)
        print(f"Transcription: {transcription}")

        # Add a small delay to simulate real-time processing
        time.sleep(chunk_duration_ms / 1000)


if __name__ == "__main__":
    stream_audio_from_file(audio_file)
