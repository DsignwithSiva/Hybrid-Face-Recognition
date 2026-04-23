import os
import cv2
import uuid
import time

from facenet_pytorch import MTCNN   # ✅ FIXED

from config import (
    VIDEO_PATH, VIDEO_PATHS, VIDEO_NAMESPACE,
    BASE_FRAME_SKIP, MIN_FACE_SIZE, MAX_FACE_SIZE,
    TRACKING_FRAME_WINDOW, MAX_FACES_TO_COLLECT,
    GPU_BATCH_SIZE, DEVICE
)

from utils import FaceTracker, BatchFaceEncoder, check_face_quality
from models import model, index

# Initialize detector
mtcnn = MTCNN(keep_all=True, device=DEVICE)


# ===============================
# STORE ALL FACES
# ===============================

def store_all_faces_from_video():

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    faces_collected = 0

    tracker = FaceTracker(frame_window=TRACKING_FRAME_WINDOW)
    batch_encoder = BatchFaceEncoder(model, DEVICE, GPU_BATCH_SIZE)

    batch_vectors = []
    BATCH_SIZE = 100

    print("🚀 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % BASE_FRAME_SKIP != 0:
            continue

        # ✅ FACE DETECTION (MTCNN)
        boxes, _ = mtcnn.detect(frame)

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face_w, face_h = x2 - x1, y2 - y1

            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                continue

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box_tuple = (x1, y1, x2, y2)

            if tracker.is_duplicate(frame_count, box_tuple):
                continue

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            quality_ok, _, confidence = check_face_quality(face_rgb)

            if not quality_ok:
                continue

            unique_id = str(uuid.uuid4())
            tracker.add_face(frame_count, box_tuple, unique_id)

            metadata = {
                "id": unique_id,
                "frame": frame_count,
                "timestamp": float(frame_count / fps),
                "video": VIDEO_PATH,
                "quality_confidence": float(confidence)
            }

            batch_encoder.add_face(face_rgb, metadata)

            encoded = batch_encoder.process_batch(force=False)

            for emb, meta in encoded:
                faces_collected += 1
                batch_vectors.append((meta["id"], emb.tolist(), meta))

                if len(batch_vectors) >= BATCH_SIZE:
                    index.upsert(vectors=batch_vectors, namespace=VIDEO_NAMESPACE)
                    batch_vectors = []

            if faces_collected >= MAX_FACES_TO_COLLECT:
                break

        if faces_collected >= MAX_FACES_TO_COLLECT:
            break

    # flush
    encoded = batch_encoder.flush()

    for emb, meta in encoded:
        batch_vectors.append((meta["id"], emb.tolist(), meta))

    if batch_vectors:
        index.upsert(vectors=batch_vectors, namespace=VIDEO_NAMESPACE)

    cap.release()

    print(f"✅ Stored {faces_collected} faces")