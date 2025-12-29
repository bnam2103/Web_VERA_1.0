# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from time import time, perf_counter
import asyncio
import numpy as np
import io
import json
import string

from intent import is_command
from ASR import transcribe_long
from LLM import VeraAI
from TTS import speak_to_file
from pydub import AudioSegment

# =========================
# CONFIG
# =========================

MODEL_PATH = r"C:\Users\User\Documents\Fine_Tuning_Projects\LLAMA_LLM_3B"

MAX_ACTIVE_USERS = 10
SESSION_TTL = 30 * 60
MAX_TURNS = 20

TARGET_SR = 16000
MIN_AUDIO_BYTES = 1500
MIN_AUDIO_RMS = 0.050  # 🔑 ENERGY GATE (MATCHES FRONTEND)
MIN_VOICED_SECONDS = 0.05   # 🔑 NEW
ZCR_MIN = 0.01
ZCR_MAX = 0.40            # 🔑 NEW (TUNER; fast speech fails with lower number)

MAX_FEEDBACK_BYTES = 1 * 1024 * 1024  # 1 MB

# =========================
# APP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS & LOCKS
# =========================

vera = VeraAI(MODEL_PATH)

asr_lock = asyncio.Lock()
llm_lock = asyncio.Lock()
tts_lock = asyncio.Lock()

# =========================
# SESSION STATE
# =========================

user_histories = defaultdict(list)
user_last_seen = {}
total_sessions_seen = set()
paused_sessions = set()

# =========================
# HELPERS
# =========================

def safe_id(value: str) -> str:
    return "".join(c for c in value if c.isalnum() or c in ("_", "-"))

def today():
    return datetime.now().strftime("%Y-%m-%d")

def timestamp():
    return datetime.now().strftime("%H-%M-%S")

def cleanup_sessions():
    now = time()
    expired = [
        sid for sid, last in user_last_seen.items()
        if now - last > SESSION_TTL
    ]
    for sid in expired:
        user_histories.pop(sid, None)
        user_last_seen.pop(sid, None)
        paused_sessions.discard(sid)

def zero_crossing_rate(samples: np.ndarray) -> float:
    return np.mean(samples[:-1] * samples[1:] < 0)

def voiced_duration(samples: np.ndarray, sr: int, thresh: float) -> float:
    mask = np.abs(samples) > thresh
    if not mask.any():
        return 0.0
    idx = np.where(mask)[0]
    return (idx[-1] - idx[0]) / sr
# =========================
# SIMPLE INTENTS
# =========================

def check_time():
    return f"The current time is {datetime.now().strftime('%I:%M %p')}."

def check_date():
    return f"Today's date is {datetime.now().strftime('%A, %B %d, %Y')}."

def detect_intent(text: str, history) -> str | None:
    # cleaned = text.lower().translate(
    #     str.maketrans("", "", string.punctuation)
    # )
    messages = build_messages(history, text)
    reply = vera.generate(messages).strip()

    if any(k in reply for k in ["current time", "realtime", "real time","real-time"]):
        return "time"
    if any(k in reply for k in ["current date", "current year", "current month"]):
        return "date"
    return None

# =========================
# PROMPT BUILDER
# =========================

def build_messages(history, user_text):
    system = (
        vera.base_system_prompt
        + "\n\nYou are VERA, a calm professional voice assistant."
        + "\nDo not use markdown, emojis, or formatting."
        + "\nYour output will be spoken aloud."
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages

# =========================
# AUDIO PATHS
# =========================

def user_tts_dir(session_id):
    p = Path("tts_outputs") / session_id / today()
    p.mkdir(parents=True, exist_ok=True)
    return p

def user_feedback_dir(session_id):
    p = Path("feedback") / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

# =========================
# METRICS LOGGER
# =========================

async def log_metrics():
    while True:
        cleanup_sessions()
        print(
            f"[METRICS] users={len(user_last_seen)}/{MAX_ACTIVE_USERS} | "
            f"ASR={asr_lock.locked()} LLM={llm_lock.locked()} TTS={tts_lock.locked()}"
        )
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup():
    asyncio.create_task(log_metrics())

# =========================
# INFERENCE
# =========================

@app.post("/infer")
async def infer(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
):
    t_start = perf_counter()

    session_id = safe_id(session_id)
    cleanup_sessions()

    if session_id not in user_last_seen and len(user_last_seen) >= MAX_ACTIVE_USERS:
        raise HTTPException(429, "Server at capacity")

    user_last_seen[session_id] = time()
    total_sessions_seen.add(session_id)

    audio_bytes = await audio.read()
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        return {"skip": True}

    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception:
        raise HTTPException(400, "Invalid audio format")

    seg = seg.set_channels(1).set_frame_rate(TARGET_SR)

    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)

    # robust normalization (always correct)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak

    # =========================
    # 🔑 ENERGY GATE (BEFORE NORMALIZATION)
    # =========================

    rms = np.sqrt(np.mean(samples ** 2))
    zcr = zero_crossing_rate(samples)
    voiced_sec = voiced_duration(samples, TARGET_SR, thresh=0.02)

    print(
        f"[DEBUG] rms={rms:.4f} "
        f"zcr={zcr:.3f} "
        f"voiced={voiced_sec:.3f}s"
    )

    if (
        rms < MIN_AUDIO_RMS or
        zcr < ZCR_MIN or zcr > ZCR_MAX or
        voiced_sec < MIN_VOICED_SECONDS
    ):
        print("[AUDIO] Dropped noise")
        return {"skip": True}

    # remove DC offset
    samples -= np.mean(samples)

    # normalize AFTER gating
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak

    # =========================
    # ASR
    # =========================

    async with asr_lock:
        transcript, confidence = transcribe_long(samples)

        print(f"[ASR] conf={confidence:.3f} text=\"{transcript}\"")

        if confidence < -0.5: #(TUNER)
            print("[ASR] Dropped low-confidence transcription")
            return {"skip": True}

    if not transcript:
        return {"skip": True}

    print(f"[ASR] \"{transcript}\"")

    # =========================
    # COMMAND HANDLING
    # =========================

    if is_command(transcript):
        t = transcript.lower()

        if "pause" in t and "unpause" not in t:
            paused_sessions.add(session_id)
            return {"command": "pause"}

        if "unpause" in t:
            paused_sessions.discard(session_id)
            return {"command": "unpause"}

    # =========================
    # PAUSED MODE
    # =========================

    if session_id in paused_sessions:
        return {"paused": True, "transcript": transcript}

    history = user_histories[session_id]

    # =========================
    # LLM
    # =========================

    intent = detect_intent(transcript, history)

    if intent == "time":
        reply = check_time()
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})
    elif intent == "date":
        reply = check_date()
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})
    else:
        messages = build_messages(history, transcript)
        async with llm_lock:
            reply = vera.generate(messages).strip()

        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})

        if len(history) > MAX_TURNS * 2:
            history[:] = history[-MAX_TURNS * 2:]

    # =========================
    # TTS
    # =========================

    tts_dir = user_tts_dir(session_id)
    fname = f"{timestamp()}.wav"
    path = tts_dir / fname

    async with tts_lock:
        speak_to_file(reply, path)

    print(f"[LATENCY] TOTAL={perf_counter() - t_start:.3f}s")

    return {
        "transcript": transcript,
        "reply": reply,
        "audio_url": f"/audio/{session_id}/{today()}/{fname}"
    }

# =========================
# AUDIO SERVING
# =========================

@app.get("/audio/{session_id}/{date}/{filename}")
def get_audio(session_id: str, date: str, filename: str):
    path = Path("tts_outputs") / safe_id(session_id) / date / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")

# =========================
# HEALTH & METRICS
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    cleanup_sessions()
    return {
        "active_users": len(user_last_seen),
        "total_sessions_seen": len(total_sessions_seen),
    }

# =========================
# FEEDBACK
# =========================

class Feedback(BaseModel):
    session_id: str
    feedback: str
    userAgent: str | None = None
    timestamp: str | None = None

@app.post("/feedback")
async def receive_feedback(data: Feedback):
    size = len(data.feedback.encode("utf-8"))
    if size > MAX_FEEDBACK_BYTES:
        raise HTTPException(413, "Feedback exceeds 1MB limit")

    path = user_feedback_dir(safe_id(data.session_id)) / "feedback.jsonl"

    entry = data.dict()
    entry["timestamp"] = entry.get("timestamp") or datetime.utcnow().isoformat()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "ok"}
