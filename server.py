"""
Hybrid Face Recognition — FastAPI Backend
Run with:  python run.py
           OR  uvicorn server:app --reload --port 8000
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

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─── TensorFlow __version__ shim (fixes broken TF installs on Windows) ──────
try:
    import tensorflow as _tf
    if not hasattr(_tf, '__version__'):
        _tf.__version__ = '2.15.0'
except Exception:
    pass

from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

# ─── App ────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Face Recognition API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───────────────────────────────────────────
_jobs: dict = {}
_proc_lock = threading.Lock()          # only one heavy job at a time
_models_loaded = False
_import_lock   = threading.Lock()

# ─── Lazy model / module import ─────────────────────────────
def _ensure_models():
    global _models_loaded
    with _import_lock:
        if _models_loaded:
            return
        global _store_modes, _search_modes, _model_module
        import store_modes   as _store_modes
        import search_modes  as _search_modes
        import models        as _model_module
        _models_loaded = True


# ─── Job helpers ────────────────────────────────────────────
def _new_job() -> str:
    jid = str(uuid.uuid4())
    _jobs[jid] = {
        "q":      queue.Queue(maxsize=2000),
        "status": "running",
        "result": None,
    }
    return jid


def _emit(jid: str, msg_type: str, **kwargs):
    if jid not in _jobs:
        return
    try:
        _jobs[jid]["q"].put_nowait({"type": msg_type, **kwargs})
    except queue.Full:
        pass


def _job_done(jid: str, result: dict):
    _jobs[jid]["status"] = "done"
    _jobs[jid]["result"] = result
    _emit(jid, "done", result=result)


def _job_error(jid: str, message: str):
    _jobs[jid]["status"] = "error"
    _emit(jid, "error", message=message)


# ─── Stdout Capture → SSE ───────────────────────────────────
class _SSECapture:
    """Redirect print() output to the SSE job queue."""
    def __init__(self, jid: str):
        self._jid = jid
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                _emit(self._jid, "log", text=stripped)
                # Parse inline progress percentage  "Progress: 45.1%"
                m = re.search(r"Progress:\s*(\d+\.?\d*)%", stripped)
                if m:
                    _emit(self._jid, "progress", value=float(m.group(1)))

    def flush(self):
        if self._buf.strip():
            _emit(self._jid, "log", text=self._buf.strip())
            self._buf = ""


@contextlib.contextmanager
def _capture(jid: str):
    old = sys.stdout
    sys.stdout = _SSECapture(jid)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = old


@contextlib.contextmanager
def _auto_input(response: str = "o"):
    """Replace input() with a fixed response for non-interactive server mode."""
    orig = builtins.input
    builtins.input = lambda *_: response
    try:
        yield
    finally:
        builtins.input = orig


def _save_upload(upload: UploadFile, suffix: str) -> str:
    """Save an upload to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# SSE STREAM ENDPOINT
# ═══════════════════════════════════════════════════════════
@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    q = _jobs[job_id]["q"]

    async def event_generator():
        last_ping = time.time()
        while True:
            try:
                msg = q.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                now = time.time()
                if now - last_ping >= 5:
                    yield 'data: {"type":"ping"}\n\n'
                    last_ping = now
                await asyncio.sleep(0.05)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════
# STATUS / NAMESPACES
# ═══════════════════════════════════════════════════════════
@app.get("/api/status")
async def get_status():
    try:
        _ensure_models()
        index = _model_module.index
        stats = index.describe_index_stats()
        namespaces = {
            ns: data.get("vector_count", 0)
            for ns, data in stats.get("namespaces", {}).items()
        }
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return JSONResponse({
            "ok": True,
            "device": device,
            "namespaces": namespaces,
            "total_vectors": stats.get("total_vector_count", 0),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/namespaces")
async def get_namespaces():
    try:
        _ensure_models()
        index = _model_module.index
        stats = index.describe_index_stats()
        ns = {
            k: v.get("vector_count", 0)
            for k, v in stats.get("namespaces", {}).items()
        }
        return JSONResponse({"ok": True, "namespaces": ns})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════
# MODE 1 — STORE
# ═══════════════════════════════════════════════════════════
def _bg_store(jid: str, video_path: str, namespace: str,
              frame_skip: int, min_face: int, max_faces: int,
              gpu_batch: int, conflict: str):
    # Map form conflict setting to auto-input response
    resp_map = {"overwrite": "o", "skip": "s", "cancel": "c"}
    auto_resp = resp_map.get(conflict, "s")

    try:
        with _capture(jid), _auto_input(auto_resp):
            _emit(jid, "log", text="🔧 Loading models (first run may take ~30s)...")
            _ensure_models()
            sm = _store_modes

            # Override module-level variables used by the function
            sm.VIDEO_PATH = video_path
            sm.VIDEO_NAMESPACE = namespace
            sm.BASE_FRAME_SKIP = int(frame_skip)
            sm.MIN_FACE_SIZE = int(min_face)
            sm.MAX_FACES_TO_COLLECT = int(max_faces)
            sm.GPU_BATCH_SIZE = int(gpu_batch)

            sm.store_all_faces_from_video()
        _job_done(jid, {"mode": "store", "namespace": namespace})
    except Exception as e:
        _emit(jid, "log", text=f"❌ Exception: {e}")
        _job_error(jid, str(e))
    finally:
        _cleanup(video_path)
        _proc_lock.release()


@app.post("/api/store")
async def api_store(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    namespace: str = Form("video_default"),
    frame_skip: int = Form(30),
    min_face_size: int = Form(80),
    max_faces: int = Form(500),
    gpu_batch: int = Form(32),
    conflict: str = Form("skip"),
):
    _proc_lock.acquire()
    jid = _new_job()
    ext = os.path.splitext(video.filename or "v")[-1] or ".mp4"
    tmp = _save_upload(video, ext)
    background_tasks.add_task(
        _bg_store, jid, tmp, namespace,
        frame_skip, min_face_size, max_faces, gpu_batch, conflict
    )
    return JSONResponse({"job_id": jid})


# ═══════════════════════════════════════════════════════════
# MODE 2 — SEARCH
# ═══════════════════════════════════════════════════════════
def _bg_search(jid: str, image_path: str, namespace: str,
               threshold: float, top_k: int, cluster: int):
    try:
        with _capture(jid), _auto_input(""):
            _emit(jid, "log", text="🔧 Loading models (first run may take ~30s)...")
            _ensure_models()
            sm = _search_modes

            sm.IMAGE_PATH = image_path
            sm.VIDEO_NAMESPACE = namespace
            sm.DIST_THRESHOLD = float(threshold)
            sm.TOP_K_RESULTS = int(top_k)
            sm.TEMPORAL_CLUSTER_THRESHOLD = int(cluster)
            sm.VIDEO_PATH = ""

            sm.search_for_person_in_stored_faces()
        _job_done(jid, {"mode": "search", "namespace": namespace})
    except Exception as e:
        _emit(jid, "log", text=f"❌ Exception: {e}")
        _job_error(jid, str(e))
    finally:
        _cleanup(image_path)
        _proc_lock.release()


@app.post("/api/search")
async def api_search(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    namespace: str = Form("video_peop"),
    threshold: float = Form(0.50),
    top_k: int = Form(100),
    cluster: int = Form(30),
):
    _proc_lock.acquire()
    jid = _new_job()
    ext = os.path.splitext(image.filename or "i")[-1] or ".jpg"
    tmp = _save_upload(image, ext)
    background_tasks.add_task(
        _bg_search, jid, tmp, namespace, threshold, top_k, cluster
    )
    return JSONResponse({"job_id": jid})


# ═══════════════════════════════════════════════════════════
# MODE 3 — BATCH SEARCH (Multiple people, one namespace)
# ═══════════════════════════════════════════════════════════
def _bg_batch_search(jid: str, image_paths: List[str], orig_names: List[str],
                     namespace: str, threshold: float, top_k: int, cluster: int):
    try:
        with _capture(jid), _auto_input(""):
            _emit(jid, "log", text="🔧 Loading models (first run may take ~30s)...")
            _ensure_models()
            sm = _search_modes

            sm.BATCH_IMAGE_PATHS          = image_paths
            sm.BATCH_IMAGE_NAMES          = orig_names
            sm.VIDEO_NAMESPACE            = namespace
            sm.DIST_THRESHOLD             = float(threshold)
            sm.TOP_K_RESULTS              = int(top_k)
            sm.TEMPORAL_CLUSTER_THRESHOLD = int(cluster)

            sm.batch_search_multiple_people()
        _job_done(jid, {"mode": "batch_search", "namespace": namespace})
    except Exception as e:
        _emit(jid, "log", text=f"❌ Exception: {e}")
        _job_error(jid, str(e))
    finally:
        _cleanup(*image_paths)
        _proc_lock.release()


@app.post("/api/batch-search")
async def api_batch_search(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    namespace: str = Form("video_peop"),
    threshold: float = Form(0.50),
    top_k: int = Form(100),
    cluster: int = Form(30),
):
    if not images:
        raise HTTPException(400, "No images provided")
    _proc_lock.acquire()
    jid = _new_job()
    tmp_paths = []
    orig_names = []
    for img in images:
        ext = os.path.splitext(img.filename or "i")[-1] or ".jpg"
        tmp_paths.append(_save_upload(img, ext))
        orig_names.append(img.filename or f"image_{len(orig_names)+1}{ext}")
    background_tasks.add_task(
        _bg_batch_search, jid, tmp_paths, orig_names, namespace, threshold, top_k, cluster
    )
    return JSONResponse({"job_id": jid})


# ═══════════════════════════════════════════════════════════
# MODE 4 — MULTI-VIDEO SEARCH (One person, many namespaces)
# ═══════════════════════════════════════════════════════════
def _bg_multi_video(jid: str, image_path: str, video_names: List[str],
                    threshold: float, top_k: int, cluster: int):
    """
    video_names: list of filenames like 'bahu_480.mp4'
    Derives namespace: video_bahu_480
    """
    try:
        with _capture(jid), _auto_input(""):
            _emit(jid, "log", text="🔧 Loading models (first run may take ~30s)...")
            _ensure_models()
            sm = _search_modes

            sm.IMAGE_PATH = image_path
            sm.VIDEO_PATHS = video_names
            sm.DIST_THRESHOLD = float(threshold)
            sm.TOP_K_RESULTS = int(top_k)
            sm.TEMPORAL_CLUSTER_THRESHOLD = int(cluster)

            sm.multi_video_search_one_person()
        _job_done(jid, {"mode": "multi_video_search", "videos": video_names})
    except Exception as e:
        _emit(jid, "log", text=f"❌ Exception: {e}")
        _job_error(jid, str(e))
    finally:
        _cleanup(image_path)
        _proc_lock.release()


@app.post("/api/multi-video-search")
async def api_multi_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    video_names: str = Form("[]"),   # JSON array of filenames/namespaces
    threshold: float = Form(0.50),
    top_k: int = Form(100),
    cluster: int = Form(30),
):
    try:
        names = json.loads(video_names)
    except Exception:
        names = [n.strip() for n in video_names.split(",") if n.strip()]
    if not names:
        raise HTTPException(400, "No video names provided")
    _proc_lock.acquire()
    jid = _new_job()
    ext = os.path.splitext(image.filename or "i")[-1] or ".jpg"
    tmp = _save_upload(image, ext)
    background_tasks.add_task(
        _bg_multi_video, jid, tmp, names, threshold, top_k, cluster
    )
    return JSONResponse({"job_id": jid})


# ═══════════════════════════════════════════════════════════
# MODE 5 — ULTIMATE SEARCH (Many people × many namespaces)
# ═══════════════════════════════════════════════════════════
def _bg_ultimate(jid: str, image_paths: List[str], video_names: List[str],
                 threshold: float, top_k: int, cluster: int):
    try:
        with _capture(jid), _auto_input(""):
            _emit(jid, "log", text="🔧 Loading models (first run may take ~30s)...")
            _ensure_models()
            sm = _search_modes

            sm.BATCH_IMAGE_PATHS = image_paths
            sm.VIDEO_PATHS = video_names
            sm.DIST_THRESHOLD = float(threshold)
            sm.TOP_K_RESULTS = int(top_k)
            sm.TEMPORAL_CLUSTER_THRESHOLD = int(cluster)

            sm.ultimate_search()
        _job_done(jid, {"mode": "ultimate_search"})
    except Exception as e:
        _emit(jid, "log", text=f"❌ Exception: {e}")
        _job_error(jid, str(e))
    finally:
        _cleanup(*image_paths)
        _proc_lock.release()


@app.post("/api/ultimate-search")
async def api_ultimate(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    video_names: str = Form("[]"),   # JSON array
    threshold: float = Form(0.50),
    top_k: int = Form(100),
    cluster: int = Form(30),
):
    try:
        names = json.loads(video_names)
    except Exception:
        names = [n.strip() for n in video_names.split(",") if n.strip()]
    if not names:
        raise HTTPException(400, "No video names provided")
    if not images:
        raise HTTPException(400, "No images provided")
    _proc_lock.acquire()
    jid = _new_job()
    tmp_paths = []
    for img in images:
        ext = os.path.splitext(img.filename or "i")[-1] or ".jpg"
        tmp_paths.append(_save_upload(img, ext))
    background_tasks.add_task(
        _bg_ultimate, jid, tmp_paths, names, threshold, top_k, cluster
    )
    return JSONResponse({"job_id": jid})


# ═══════════════════════════════════════════════════════════
# MODE 6 — BULK STORE (Many videos → separate namespaces)
# ═══════════════════════════════════════════════════════════
def _bg_bulk_store(jid: str, video_paths: List[str]):
    try:
        with _capture(jid), _auto_input("s"):   # 's' = skip existing
            _emit(jid, "log", text="🔧 Loading models (first run may take ~30s)...")
            _ensure_models()
            sm = _store_modes

            sm.VIDEO_PATHS = video_paths

            sm.bulk_store_multiple_videos()
        _job_done(jid, {"mode": "bulk_store", "videos": len(video_paths)})
    except Exception as e:
        _emit(jid, "log", text=f"❌ Exception: {e}")
        _job_error(jid, str(e))
    finally:
        _cleanup(*video_paths)
        _proc_lock.release()


@app.post("/api/bulk-store")
async def api_bulk_store(
    background_tasks: BackgroundTasks,
    videos: List[UploadFile] = File(...),
):
    if not videos:
        raise HTTPException(400, "No videos provided")
    _proc_lock.acquire()
    jid = _new_job()
    tmp_paths = []
    for v in videos:
        ext = os.path.splitext(v.filename or "v")[-1] or ".mp4"
        tmp_paths.append(_save_upload(v, ext))
    background_tasks.add_task(_bg_bulk_store, jid, tmp_paths)
    return JSONResponse({"job_id": jid})


# ═══════════════════════════════════════════════════════════
# SERVE FRONTEND (must be last)
# ═══════════════════════════════════════════════════════════
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# ─── Direct Run ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
