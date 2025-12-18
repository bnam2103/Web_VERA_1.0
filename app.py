# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from time import time
import asyncio
import numpy as np
import io
import json
import string

from ASR import transcribe_long
from LLM import VeraAI
from TTS import speak_to_file
from pydub import AudioSegment

# =========================
# CONFIG
# =========================

MODEL_PATH = r"C:\Users\User\Documents\Fine_Tuning_Projects\LLAMA_LLM_3B"
MAX_ACTIVE_USERS = 10
SESSION_TTL = 30 * 60  # 30 minutes
MAX_TURNS = 20

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
# GLOBAL MODEL & GPU LOCK
# =========================

vera = VeraAI(MODEL_PATH)
gpu_lock = asyncio.Lock()

# =========================
# SESSION STATE (UUID ONLY)
# =========================

user_histories = defaultdict(list)   # session_id -> chat turns
user_last_seen = {}
total_sessions_seen = set()

# =========================
# BASIC HELPERS
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

# =========================
# TIME / DATE RESPONSES
# =========================

def check_time():
    now = datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}."

def check_date():
    today_dt = datetime.now()
    return f"Today's date is {today_dt.strftime('%A, %B %d, %Y')}."

# =========================
# PROMPT BUILDER (FINAL)
# =========================

def build_messages(history: list | None, user_text: str):
    """
    - Stable VERA identity
    - No user identity
    - Creator info only when explicitly asked
    """

    if history is None:
        history = []

    user_text_l = user_text.lower()

    creator_triggers = [
        "who made you",
        "who created you",
        "your creator",
        "your origin",
        "who built you"
    ]

    creator_allowed = any(t in user_text_l for t in creator_triggers)

    system_content = (
        vera.base_system_prompt
        + "\n\nIMPORTANT RULES:\n"
          "You are VERA, a conversational voice-based AI.\n"
          "Speak calmly, professionally, and concisely.\n"
          "Do not use markdown, emojis, or formatting.\n"
          "Your output will be spoken aloud.\n\n"
    )

    if creator_allowed:
        system_content += (
            "\nCREATOR INFORMATION:\n"
            + vera.creator_info + "\n"
        )
    else:
        system_content += (
            "\nDo NOT mention or reference your creator unless explicitly asked.\n"
        )

    messages = [{"role": "system", "content": system_content}]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": user_text
    })

    return messages

# =========================
# INTENT CHECK (LLM-BASED)
# =========================

def is_time_or_date_query(history, text: str) -> str | None:
    """
    Uses LLM phrasing to normalize many time/date question variants.
    """
    cleaned = text.lower().translate(
        str.maketrans("", "", string.punctuation)
    )

    messages = build_messages(history, cleaned)
    normalized = vera.generate(messages).lower()

    if "current time" in normalized:
        return "time"

    if "current date" in normalized:
        return "date"

    return None

# =========================
# AUDIO PATHS (UUID ONLY)
# =========================

def user_audio_dir(session_id):
    p = Path("audio_logs") / session_id / today()
    p.mkdir(parents=True, exist_ok=True)
    return p

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

async def log_metrics_periodically():
    while True:
        cleanup_sessions()
        print(
            f"[METRICS] Active users: {len(user_last_seen)}/{MAX_ACTIVE_USERS} | "
            f"GPU: {'BUSY' if gpu_lock.locked() else 'IDLE'}"
        )
        await asyncio.sleep(10)

@app.on_event("startup")
async def start_metrics_logger():
    asyncio.create_task(log_metrics_periodically())

# =========================
# INFERENCE
# =========================

@app.post("/infer")
async def infer(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
):
    session_id = safe_id(session_id)
    cleanup_sessions()

    if session_id not in user_last_seen and len(user_last_seen) >= MAX_ACTIVE_USERS:
        raise HTTPException(429, "VERA is currently at capacity.")

    user_last_seen[session_id] = time()
    total_sessions_seen.add(session_id)

    audio_bytes = await audio.read()
    audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)

    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

    async with gpu_lock:

        transcript = transcribe_long(samples).strip()

        history = user_histories[session_id]

        if not transcript:
            reply = "I didn’t catch that. Could you try again?"
        else:
            intent = is_time_or_date_query(history, transcript)

            if intent == "time":
                reply = check_time()

            elif intent == "date":
                reply = check_date()

            else:
                messages = build_messages(history, transcript)
                reply = vera.generate(messages).strip()

                history.append({"role": "user", "content": transcript})
                history.append({"role": "assistant", "content": reply})

                if len(history) > MAX_TURNS * 2:
                    history[:] = history[-MAX_TURNS * 2:]

        tts_dir = user_tts_dir(session_id)
        fname = f"{timestamp()}.wav"
        tts_path = tts_dir / fname

        speak_to_file(reply, tts_path)

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
        "gpu_busy": gpu_lock.locked(),
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
    path = user_feedback_dir(safe_id(data.session_id)) / "feedback.jsonl"

    entry = data.dict()
    entry["timestamp"] = entry.get("timestamp") or datetime.utcnow().isoformat()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "ok"}
