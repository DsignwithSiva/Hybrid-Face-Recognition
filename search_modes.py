import os
import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN

from config import (
    IMAGE_PATH, VIDEO_PATH, VIDEO_PATHS, VIDEO_NAMESPACE,
    BATCH_IMAGE_PATHS, DIST_THRESHOLD, TEMPORAL_CLUSTER_THRESHOLD,
    TOP_K_RESULTS, DEVICE
)

from utils import l2_normalize, TemporalClusterer
from models import model, index

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device=DEVICE)

BATCH_IMAGE_NAMES: list = []


# ===============================
# ENCODE REFERENCE IMAGE
# ===============================

def encode_reference_image(image_path: str):
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(ref_img_rgb)

    if boxes is None:
        raise ValueError(f"No faces detected in: {image_path}")

    # Select largest face
    largest_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1, y1, x2, y2 = map(int, largest_box)

    face = ref_img_rgb[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))

    ref_tensor = (
        torch.tensor(face)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0) / 255.0
    ).to(DEVICE)

    with torch.no_grad():
        emb = model(ref_tensor).cpu().numpy()[0]

    return l2_normalize(emb)


# ===============================
# SEARCH SINGLE PERSON
# ===============================

def search_for_person_in_stored_faces():

    ref_emb = encode_reference_image(IMAGE_PATH)

    results = index.query(
        vector=ref_emb.tolist(),
        top_k=TOP_K_RESULTS,
        include_metadata=True,
        namespace=VIDEO_NAMESPACE
    )

    matches = []
    clusterer = TemporalClusterer(frame_threshold=TEMPORAL_CLUSTER_THRESHOLD)

    for match in results['matches']:
        distance = 1 - match['score']

        if distance < DIST_THRESHOLD:
            confidence = match['metadata'].get('quality_confidence', 0.5)
            frame_num = match['metadata']['frame']

            clusterer.add_detection(frame_num, distance, confidence)

            matches.append({
                'frame': frame_num,
                'distance': distance
            })

    clusters = clusterer.get_clusters()

    if len(clusters) > 0:
        print("✅ Person FOUND")
    else:
        print("❌ Person NOT FOUND")


# ===============================
# BATCH SEARCH
# ===============================

def batch_search_multiple_people():

    all_results = {}

    for image_path in BATCH_IMAGE_PATHS:

        try:
            ref_emb = encode_reference_image(image_path)
        except Exception as e:
            all_results[image_path] = {"status": "error", "error": str(e)}
            continue

        results = index.query(
            vector=ref_emb.tolist(),
            top_k=TOP_K_RESULTS,
            include_metadata=True,
            namespace=VIDEO_NAMESPACE
        )

        matches = []
        clusterer = TemporalClusterer(frame_threshold=TEMPORAL_CLUSTER_THRESHOLD)

        for match in results['matches']:
            distance = 1 - match['score']

            if distance < DIST_THRESHOLD:
                frame_num = match['metadata']['frame']
                clusterer.add_detection(frame_num, distance, 0.5)
                matches.append(frame_num)

        clusters = clusterer.get_clusters()

        all_results[image_path] = {
            "status": "found" if len(clusters) > 0 else "not_found",
            "matches": len(matches)
        }

    print(all_results)


# ===============================
# MULTI VIDEO SEARCH
# ===============================

def multi_video_search_one_person():

    ref_emb = encode_reference_image(IMAGE_PATH)

    for video_path in VIDEO_PATHS:

        video_namespace = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"

        results = index.query(
            vector=ref_emb.tolist(),
            top_k=TOP_K_RESULTS,
            include_metadata=True,
            namespace=video_namespace
        )

        found = any((1 - m['score']) < DIST_THRESHOLD for m in results['matches'])

        print(f"{video_path}: {'FOUND' if found else 'NOT FOUND'}")


# ===============================
# ULTIMATE SEARCH
# ===============================

def ultimate_search():

    for image_path in BATCH_IMAGE_PATHS:

        print(f"\n👤 {image_path}")

        try:
            ref_emb = encode_reference_image(image_path)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        for video_path in VIDEO_PATHS:

            video_namespace = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"

            results = index.query(
                vector=ref_emb.tolist(),
                top_k=TOP_K_RESULTS,
                include_metadata=True,
                namespace=video_namespace
            )

            found = any((1 - m['score']) < DIST_THRESHOLD for m in results['matches'])

            print(f"   {video_path}: {'FOUND' if found else 'NOT FOUND'}")