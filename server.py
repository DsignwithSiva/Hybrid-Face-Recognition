"""
Hybrid Face Recognition — FastAPI Backend
Run with: uvicorn server:app --host 0.0.0.0 --port 10000
"""

import os
import sys
import time
import uuid
import json
import shutil
import tempfile
import asyncio
import builtins
import threading
import contextlib
import queue
import re
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

# ─── App ─────────────────────────────
app = FastAPI(title="Hybrid Face Recognition API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────
_jobs = {}
_proc_lock = threading.Lock()
_models_loaded = False
_import_lock = threading.Lock()

# ─── Lazy Load ────────────────────────
def _ensure_models():
    global _models_loaded
    with _import_lock:
        if _models_loaded:
            return
        global _store_modes, _search_modes, _model_module
        import store_modes as _store_modes
        import search_modes as _search_modes
        import models as _model_module
        _models_loaded = True


# ─── Job Helpers ──────────────────────
def _new_job():
    jid = str(uuid.uuid4())
    _jobs[jid] = {"q": queue.Queue(), "status": "running"}
    return jid

def _emit(jid, msg_type, **kwargs):
    if jid in _jobs:
        _jobs[jid]["q"].put({"type": msg_type, **kwargs})

def _job_done(jid, result):
    _jobs[jid]["status"] = "done"
    _emit(jid, "done", result=result)

def _job_error(jid, msg):
    _jobs[jid]["status"] = "error"
    _emit(jid, "error", message=msg)


# ─── Capture Logs ─────────────────────
class _Capture:
    def __init__(self, jid):
        self.jid = jid

    def write(self, text):
        text = text.strip()
        if text:
            _emit(self.jid, "log", text=text)

    def flush(self):
        pass


@contextlib.contextmanager
def _capture(jid):
    old = sys.stdout
    sys.stdout = _Capture(jid)
    try:
        yield
    finally:
        sys.stdout = old


# ─── Save Upload ──────────────────────
def _save_upload(upload, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


# ═════════════════════════════════════
# STATUS
# ═════════════════════════════════════
@app.get("/api/status")
async def get_status():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {"ok": True, "device": device}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═════════════════════════════════════
# SEARCH API
# ═════════════════════════════════════
def _bg_search(jid, image_path, namespace):
    try:
        with _capture(jid):
            _ensure_models()
            sm = _search_modes

            sm.IMAGE_PATH = image_path
            sm.VIDEO_NAMESPACE = namespace

            sm.search_for_person_in_stored_faces()

        _job_done(jid, {"status": "done"})
    except Exception as e:
        _job_error(jid, str(e))


@app.post("/api/search")
async def api_search(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    namespace: str = Form("video_default"),
):
    jid = _new_job()
    tmp = _save_upload(image, ".jpg")
    background_tasks.add_task(_bg_search, jid, tmp, namespace)
    return {"job_id": jid}


# ═════════════════════════════════════
# STREAM
# ═════════════════════════════════════
@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404)

    async def event_gen():
        q = _jobs[job_id]["q"]
        while True:
            try:
                msg = q.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except:
                await asyncio.sleep(0.1)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ═════════════════════════════════════
# FRONTEND
# ═════════════════════════════════════
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ═════════════════════════════════════
# STORE API
# ═════════════════════════════════════
def _bg_store(jid, video_path, namespace):
    try:
        with _capture(jid):
            _ensure_models()
            sm = _store_modes

            sm.VIDEO_PATH = video_path
            sm.VIDEO_NAMESPACE = namespace

            sm.store_all_faces_from_video()

        _job_done(jid, {"status": "done"})
    except Exception as e:
        _job_error(jid, str(e))


@app.post("/api/store")
async def api_store(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    namespace: str = Form("video_default"),
):
    jid = _new_job()
    tmp = _save_upload(video, ".mp4")
    background_tasks.add_task(_bg_store, jid, tmp, namespace)
    return {"job_id": jid}


# ─── Run ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

    