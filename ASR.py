import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
import tempfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# -------------------------
# CONFIG
# -------------------------

MODEL_PATH = r"C:\Users\User\Documents\Fine_Tuning_Projects\ASR_Project\ASR_model"  # update this

SAMPLE_RATE = 16000        # Whisper requires 16 kHz
DURATION = 5               # seconds to record

# -------------------------
# LOAD MODEL
# -------------------------

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"Model loaded on: {device}")


# -------------------------
# RECORD AUDIO
# -------------------------

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    # print(f"\n🎙️ Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype="float32"
    )
    sd.wait() 
    # print("Recording complete.\n")
    return audio.squeeze()

# -------------------------
# RECORD AUDIO UNTIL SILENCE
# -------------------------

def load_silero_vad_torch():
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    return vad_model, get_speech_timestamps, VADIterator

vad_model, get_speech_timestamps, VADIterator = load_silero_vad_torch()
vad_iterator = VADIterator(vad_model)

def record_until_silence(
    sample_rate=16000,
    chunk_ms=20 ,
    silence_time=1.35
):
    # print("\nListening... start speaking!")

    chunk_size = int(sample_rate * chunk_ms / 1000)
    silent_chunks_needed = int(silence_time * 1000 / chunk_ms)

    audio_buffer = []
    vad_buffer = torch.zeros(0)  # accumulate samples for VAD
    silent_count = 0

    vad_iterator.reset_states()

    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        dtype="float32",
        blocksize=chunk_size,
    ) as stream:

        while True:
            chunk, _ = stream.read(chunk_size)
            chunk = chunk.squeeze()

            # add chunk to main audio buffer
            audio_buffer.append(chunk.copy())

            # append chunk to VAD buffer
            vad_buffer = torch.cat([vad_buffer, torch.tensor(chunk, dtype=torch.float32)])

            # only run VAD when we have enough samples (>= 512)
            if vad_buffer.numel() < 512:
                continue

            # run VAD on latest ~30–50 ms window
            window = vad_buffer[-512:]

            speech_prob = vad_model(window, sample_rate).item()

            if speech_prob < 0.3:
                silent_count += 1
            else:
                silent_count = 0

            if silent_count >= silent_chunks_needed:
                break

    print("Silence detected — stop.\n")

    full_audio = np.concatenate(audio_buffer, axis=0).astype("float32")
    return full_audio
# -------------------------
# TRANSCRIBE AUDIO
# -------------------------

# Function to chunk long audio into smaller segments
def chunk_audio(audio, chunk_length=30, overlap=0, sample_rate=16000):
    chunk_samples = chunk_length * sample_rate
    overlap_samples = overlap * sample_rate

    chunks = []
    start = 0

    while start < len(audio):
        end = start + chunk_samples
        chunk = audio[start:end]

        if len(chunk) == 0:
            break

        chunks.append(chunk)
        start = end - overlap_samples  # overlap

    return chunks


def transcribe_long(audio_np):
    chunks = chunk_audio(audio_np)

    texts = []
    for i, chunk in enumerate(chunks):
        # print(f"Transcribing chunk {i+1}/{len(chunks)}")

        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        # print(text) # Debugging
        texts.append(text)

    return " ".join(texts)

# Default transcribe function
def transcribe(audio_np):
    # Whisper expects float32 audio at 16 kHz
    inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


# -------------------------
# MAIN
# -------------------------

# if __name__ == "__main__":
#     print("=== ASR Test ===")
#     audio = record_until_silence()

#     print("🔁 Transcribing...")
#     text = transcribe(audio)

#     print("\n📝 Transcription:")
#     print(text)
