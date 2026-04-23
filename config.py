import os
import argparse
import torch
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# CONFIGURATION
# -------------------------------

# HYBRID MODE SETTINGS
MODE = "search"  # store / search / batch_search / multi_video_search / ultimate_search / bulk_store

# -------------------------------
# CLI ARGUMENTS
# -------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--mode', default=None)
_parser.add_argument('--video', default=None, nargs='+')
_parser.add_argument('--image', default=None, nargs='+')
_parser.add_argument('--ns', default=None)
_args, _ = _parser.parse_known_args()

if _args.mode:
    MODE = _args.mode

# -------------------------------
# FILE PATHS
# -------------------------------

VIDEO_PATH = "test_video.mp4"
IMAGE_PATH = "nani.jpg"

BATCH_IMAGE_PATHS = _args.image if (_args.image and len(_args.image) > 1) else [
    "prabhas.jpg",
    "satya.jpg"
]

VIDEO_PATHS = _args.video if (_args.video and len(_args.video) > 1) else [
    "bahu_480.mp4",
    "120_fps.mp4",
    "bhaai.mp4"
]

# -------------------------------
# PROCESSING CONFIG
# -------------------------------

BASE_FRAME_SKIP = 30
MIN_FACE_SIZE = 80
MAX_FACE_SIZE = 800

ENABLE_QUALITY_CHECKS = False
BLUR_THRESHOLD = 50.0
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220

USE_SIMPLE_TRACKING = True
TRACKING_FRAME_WINDOW = 30
MAX_FACES_TO_COLLECT = 500

GPU_BATCH_SIZE = 16   # 🔥 reduced for Render stability

# -------------------------------
# SEARCH CONFIG
# -------------------------------

DIST_THRESHOLD = 0.50
TEMPORAL_CLUSTER_THRESHOLD = 30
TOP_K_RESULTS = 50   # 🔥 reduced for faster API

# -------------------------------
# DATABASE CONFIG
# -------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEXNAME")

VECTOR_DIM = 512

# -------------------------------
# DEVICE CONFIG
# -------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# NAMESPACE
# -------------------------------

VIDEO_NAMESPACE = _args.ns if _args.ns else f"video_{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}"