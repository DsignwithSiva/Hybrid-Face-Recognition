import os
import torch
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# CONFIGURATION
# -------------------------------

# HYBRID MODE SETTINGS
MODE = "search"  # Options: "store", "search", "batch_search", "multi_video_search", "ultimate_search", "bulk_store"

# For STORE mode - single video
VIDEO_PATH = "peop.mp4"

# For SEARCH mode - single person in single video
IMAGE_PATH = "new.jpg"

# For BATCH_SEARCH mode - multiple people in ONE video
BATCH_IMAGE_PATHS = [
    "pra.jpg",
    "satya.jpg"
]

# For MULTI_VIDEO_SEARCH mode - one person across MULTIPLE videos
VIDEO_PATHS = [
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
