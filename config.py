import os
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
import torch
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# CONFIGURATION
# -------------------------------

# HYBRID MODE SETTINGS
MODE = "store"  # Options: "store", "search", "batch_search", "multi_video_search", "ultimate_search", "bulk_store"

# ─── CLI overrides (pass --video, --image, --mode etc.) ──────
# Usage examples:
#   python main.py --mode store --video path/to/video.mp4
#   python main.py --mode search --image path/to/photo.jpg
#   python main.py --mode bulk_store --video vid1.mp4 vid2.mp4
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--mode',    default=None)
_parser.add_argument('--video',   default=None, nargs='+')
_parser.add_argument('--image',   default=None, nargs='+')
_parser.add_argument('--ns',      default=None,  help='Override namespace')
_args, _ = _parser.parse_known_args()

# Apply CLI mode override
if _args.mode:
    MODE = _args.mode

# For STORE mode - single video
VIDEO_PATH = "bahu_480.mp4"

# For SEARCH mode - single person in single video
IMAGE_PATH = "pra.jpg"

# For BATCH_SEARCH mode - multiple people in ONE video
BATCH_IMAGE_PATHS = _args.image if (_args.image and len(_args.image) > 1) else [
    "pra.jpg",
    "satya.jpg"
]

# For MULTI_VIDEO_SEARCH mode - one person across MULTIPLE videos
VIDEO_PATHS = _args.video if (_args.video and len(_args.video) > 1) else [
    "bahu_480.mp4",
    "120_fps.mp4",
    "bhaai.mp4"
]

# For ULTIMATE_SEARCH mode - MULTIPLE people across MULTIPLE videos
# Uses both BATCH_IMAGE_PATHS and VIDEO_PATHS above

# Processing Configuration
BASE_FRAME_SKIP = 30        # Process every 1 second - MORE FACES DETECTED (changed from 90)
MIN_FACE_SIZE = 80
MAX_FACE_SIZE = 800

# Quality filters
ENABLE_QUALITY_CHECKS = False
BLUR_THRESHOLD = 50.0
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220

# Tracking
USE_SIMPLE_TRACKING = True
TRACKING_FRAME_WINDOW = 30
MAX_FACES_TO_COLLECT = 500  # Increased for storing all faces

# Batch processing
GPU_BATCH_SIZE = 32         # Optimized for 8GB RAM

# Search configuration
DIST_THRESHOLD = 0.50       # If still not detecting, try 0.55 or 0.60
TEMPORAL_CLUSTER_THRESHOLD = 30
TOP_K_RESULTS = 100         # Number of results to retrieve in search

# Database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEXNAME")
VECTOR_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generate unique namespace for this video
VIDEO_NAMESPACE = f"video_{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}"
