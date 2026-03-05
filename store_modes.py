import os
import cv2
import uuid
import time

# Fix broken TensorFlow __version__ on Windows
try:
    import tensorflow as _tf
    if not hasattr(_tf, '__version__'):
        _tf.__version__ = '2.15.0'
except Exception:
    pass

from retinaface import RetinaFace

from config import (
    VIDEO_PATH, VIDEO_PATHS, VIDEO_NAMESPACE,
    BASE_FRAME_SKIP, MIN_FACE_SIZE, MAX_FACE_SIZE,
    TRACKING_FRAME_WINDOW, MAX_FACES_TO_COLLECT,
    GPU_BATCH_SIZE, DEVICE
)
from utils import FaceTracker, BatchFaceEncoder, check_face_quality
from models import model, index


# ===============================
# MODE 1: STORE ALL FACES
# ===============================

def store_all_faces_from_video():
    """
    Process video and store ALL detected faces in Pinecone.
    This only needs to be done ONCE per video.
    """
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"📊 Video Info:")
    print(f"   - Duration: {duration:.1f}s")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Namespace: {VIDEO_NAMESPACE}")
    print(f"   - Processing every {BASE_FRAME_SKIP} frames...")
    print(f"   ⏳ Starting face extraction, please wait...")

    # Check if namespace already has data
    try:
        stats = index.describe_index_stats()
        namespace_count = stats.get('namespaces', {}).get(VIDEO_NAMESPACE, {}).get('vector_count', 0)

        if namespace_count > 0:
            print(f"⚠️  WARNING: Namespace '{VIDEO_NAMESPACE}' already contains {namespace_count} faces!")
            response = input("(o)verwrite, (s)kip, or (c)ancel? [o/s/c]: ").lower()

            if response == 'c':
                cap.release()
                return
            elif response == 'o':
                index.delete(delete_all=True, namespace=VIDEO_NAMESPACE)
            elif response == 's':
                print(f"⏭️  Skipping storage, namespace already populated")
                cap.release()
                return
    except Exception:
        pass

    # Initialize processing
    frame_count = 0
    frames_processed = 0
    faces_collected = 0
    faces_skipped_quality = 0
    faces_skipped_duplicate = 0

    batch_vectors = []
    BATCH_SIZE = 100

    tracker = FaceTracker(frame_window=TRACKING_FRAME_WINDOW)
    batch_encoder = BatchFaceEncoder(model, DEVICE, GPU_BATCH_SIZE)

    start_time = time.time()
    last_progress_time = start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % BASE_FRAME_SKIP != 0:
            continue

        frames_processed += 1

        # Progress update every 5 seconds
        now = time.time()
        if now - last_progress_time >= 5:
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            elapsed = now - start_time
            print(f"Progress: {pct:.1f}% | Faces: {faces_collected} | Time: {elapsed:.0f}s")
            last_progress_time = now

        # Face detection
        detections = RetinaFace.detect_faces(frame, threshold=0.5)
        if not isinstance(detections, dict):
            continue

        for face_id, face_data in detections.items():
            x1, y1, x2, y2 = face_data["facial_area"]
            face_w, face_h = x2 - x1, y2 - y1

            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE or \
               face_w > MAX_FACE_SIZE or face_h > MAX_FACE_SIZE:
                continue

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box = (x1, y1, x2, y2)

            if tracker.is_duplicate(frame_count, box):
                faces_skipped_duplicate += 1
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            quality_ok, quality_metrics, confidence = check_face_quality(face_rgb)
            if not quality_ok:
                faces_skipped_quality += 1
                continue

            unique_id = str(uuid.uuid4())
            tracker.add_face(frame_count, box, unique_id)

            metadata = {
                "id": unique_id,
                "frame": frame_count,
                "timestamp": float(frame_count / fps),
                "video": VIDEO_PATH,
                "quality_confidence": float(confidence)
            }
            batch_encoder.add_face(face_rgb, metadata)

            encoded_faces = batch_encoder.process_batch(force=False)

            for emb, meta in encoded_faces:
                faces_collected += 1
                batch_vectors.append((meta["id"], emb.tolist(), meta))

                if len(batch_vectors) >= BATCH_SIZE:
                    index.upsert(vectors=batch_vectors, namespace=VIDEO_NAMESPACE)
                    batch_vectors = []

            if faces_collected >= MAX_FACES_TO_COLLECT:
                break

        if faces_collected >= MAX_FACES_TO_COLLECT:
            break

    # Flush remaining faces
    encoded_faces = batch_encoder.flush()
    for emb, meta in encoded_faces:
        faces_collected += 1
        batch_vectors.append((meta["id"], emb.tolist(), meta))

    if len(batch_vectors) > 0:
        index.upsert(vectors=batch_vectors, namespace=VIDEO_NAMESPACE)

    cap.release()

    processing_time = time.time() - start_time
    print(f"✅ Done — {faces_collected} vectors indexed into '{VIDEO_NAMESPACE}' in {processing_time:.1f}s")
# ===============================
# MODE 6A: BULK STORE (Store Multiple Videos - Separate Namespaces)
# ===============================

def bulk_store_multiple_videos():
    """
    Process and store MULTIPLE videos at once.
    Useful for: Building a large face database from many videos
    """
    print(f"\n🎬 Processing {len(VIDEO_PATHS)} videos...")

    bulk_results = {}
    total_start_time = time.time()

    for video_idx, video_path in enumerate(VIDEO_PATHS, 1):

        video_namespace = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"

        if not os.path.exists(video_path):
            print(f"   ❌ ERROR: Video file not found!")
            bulk_results[video_path] = {"status": "error", "message": "File not found"}
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   ❌ ERROR: Could not open video!")
            bulk_results[video_path] = {"status": "error", "message": "Could not open"}
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"📊 Video Info:")
        print(f"   - Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {total_frames}")
        print(f"   - Namespace: {video_namespace}")

        try:
            stats = index.describe_index_stats()
            namespace_count = stats.get('namespaces', {}).get(video_namespace, {}).get('vector_count', 0)

            if namespace_count > 0:
                print(f"   ⚠️  Already has {namespace_count} faces stored")
                response = input("   (o)verwrite, (s)kip, or (c)ancel all? [o/s/c]: ").lower().strip()

                if response == 'c':
                    print("   ❌ Cancelled by user - stopping bulk process")
                    cap.release()
                    return
                elif response == 's':
                    print("   ⏭️  Skipping this video")
                    bulk_results[video_path] = {"status": "skipped", "faces": namespace_count}
                    cap.release()
                    continue
                elif response == 'o':
                    print("   🗑️  Clearing existing data...")
                    index.delete(delete_all=True, namespace=video_namespace)
        except Exception as e:
            print(f"   ℹ️  Could not check namespace: {e}")

        frame_count = 0
        faces_collected = 0
        faces_skipped = 0
        batch_vectors = []
        BATCH_SIZE = 100

        tracker = FaceTracker(frame_window=TRACKING_FRAME_WINDOW)
        batch_encoder = BatchFaceEncoder(model, DEVICE, GPU_BATCH_SIZE)

        video_start_time = time.time()

        print(f"\n   🔄 Extracting faces...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % BASE_FRAME_SKIP != 0:
                continue

            if frame_count % max(1, total_frames // 5) == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - video_start_time
                print(f"      {progress:.0f}% | Faces: {faces_collected} | Time: {elapsed:.0f}s")

            try:
                detections = RetinaFace.detect_faces(frame, threshold=0.5)

                if isinstance(detections, dict):
                    for face_id, face_data in detections.items():
                        x1, y1, x2, y2 = face_data["facial_area"]
                        face_w, face_h = x2 - x1, y2 - y1

                        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                            faces_skipped += 1
                            continue

                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        box = (x1, y1, x2, y2)

                        if tracker.is_duplicate(frame_count, box):
                            faces_skipped += 1
                            continue

                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue

                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                        quality_ok, _, confidence = check_face_quality(face_rgb)
                        if not quality_ok:
                            faces_skipped += 1
                            continue

                        unique_id = str(uuid.uuid4())
                        tracker.add_face(frame_count, box, unique_id)

                        metadata = {
                            "id": unique_id,
                            "frame": frame_count,
                            "timestamp": float(frame_count / fps),
                            "video": video_path,
                            "quality_confidence": float(confidence)
                        }

                        batch_encoder.add_face(face_rgb, metadata)

                        encoded = batch_encoder.process_batch(force=False)
                        for emb, meta in encoded:
                            faces_collected += 1
                            batch_vectors.append((meta["id"], emb.tolist(), meta))

                            if len(batch_vectors) >= BATCH_SIZE:
                                index.upsert(vectors=batch_vectors, namespace=video_namespace)
                                batch_vectors = []

                        if faces_collected >= MAX_FACES_TO_COLLECT:
                            break
            except Exception as e:
                print(f"      ⚠️  Frame {frame_count} error: {e}")

            if faces_collected >= MAX_FACES_TO_COLLECT:
                break

        # Flush remaining
        encoded = batch_encoder.flush()
        for emb, meta in encoded:
            faces_collected += 1
            batch_vectors.append((meta["id"], emb.tolist(), meta))

        if batch_vectors:
            index.upsert(vectors=batch_vectors, namespace=video_namespace)

        cap.release()

        video_time = time.time() - video_start_time

        bulk_results[video_path] = {
            "status": "success",
            "faces": faces_collected,
            "skipped": faces_skipped,
            "time": video_time,
            "namespace": video_namespace
        }

        print(f"\n   ✅ Stored {faces_collected} faces in {video_time:.1f}s")

    total_time = time.time() - total_start_time

    print(f"\n{'='*70}")
    print(f"📊 BULK STORAGE SUMMARY")
    print(f"{'='*70}\n")

    success_count = sum(1 for r in bulk_results.values() if r["status"] == "success")
    total_faces = sum(r.get("faces", 0) for r in bulk_results.values() if r["status"] == "success")

    print(f"{'Video':<30} | {'Status':<10} | {'Faces':<10} | {'Time'}")
    print(f"{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for video_path, result in bulk_results.items():
        video_name = os.path.basename(video_path)
        status = result["status"]

        if status == "success":
            print(f"{video_name:<30} | {'✅ Success':<10} | {result['faces']:<10} | {result['time']:.1f}s")
        elif status == "skipped":
            print(f"{video_name:<30} | {'⏭️  Skipped':<10} | {result.get('faces', 0):<10} | -")
        else:
            print(f"{video_name:<30} | {'❌ Error':<10} | {'-':<10} | -")

    print(f"\n{'='*70}")
    print(f"✅ Processed: {success_count}/{len(VIDEO_PATHS)} videos")
    print(f"📊 Total faces stored: {total_faces:,}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"{'='*70}")


# ===============================
# MODE 6B: BULK STORE (Store Multiple Videos in ONE Namespace)
# ===============================

def bulk_store_multiple_videos_single_namespace():
    """
    Process and store MULTIPLE videos in ONE shared namespace.
    All faces from all videos go into the same database for unified searching.
    """

    BULK_NAMESPACE = "bulk_videos_combined"

    print(f"\n🎬 Processing {len(VIDEO_PATHS)} videos into namespace: {BULK_NAMESPACE}")

    try:
        stats = index.describe_index_stats()
        namespace_count = stats.get('namespaces', {}).get(BULK_NAMESPACE, {}).get('vector_count', 0)

        if namespace_count > 0:
            print(f"\n⚠️  WARNING: Namespace '{BULK_NAMESPACE}' already contains {namespace_count} faces!")
            response = input("   Do you want to (o)verwrite, (a)ppend, or (c)ancel? [o/a/c]: ").lower().strip()

            if response == 'c':
                print("   ❌ Cancelled by user")
                return
            elif response == 'o':
                print(f"   🗑️  Clearing existing data...")
                index.delete(delete_all=True, namespace=BULK_NAMESPACE)
                print(f"   ✅ Namespace cleared")
            elif response == 'a':
                print(f"   ➕ Appending to existing {namespace_count} faces")
            else:
                print(f"   Invalid input, cancelling...")
                return
    except Exception as e:
        print(f"   ℹ️  Namespace is new or empty: {e}")

    bulk_results = {}
    failed_videos = []
    total_start_time = time.time()

    batch_vectors = []
    BATCH_SIZE = 100
    batch_encoder = BatchFaceEncoder(model, DEVICE, GPU_BATCH_SIZE)

    total_faces_collected = 0
    total_faces_skipped = 0

    for video_idx, video_path in enumerate(VIDEO_PATHS, 1):

        if not os.path.exists(video_path):
            error_msg = f"File not found: {video_path}"
            print(f"   ❌ ERROR: {error_msg}")
            failed_videos.append({"video": video_path, "error": error_msg})
            bulk_results[video_path] = {"status": "error", "error": error_msg}
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video: {video_path}"
            print(f"   ❌ ERROR: {error_msg}")
            cap.release()
            failed_videos.append({"video": video_path, "error": error_msg})
            bulk_results[video_path] = {"status": "error", "error": error_msg}
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        if total_frames == 0 or fps == 0:
            error_msg = f"Invalid video properties: {video_path}"
            print(f"   ❌ ERROR: {error_msg}")
            cap.release()
            failed_videos.append({"video": video_path, "error": error_msg})
            bulk_results[video_path] = {"status": "error", "error": error_msg}
            continue

        print(f"📊 Video Info:")
        print(f"   - Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {total_frames}")

        frame_count = 0
        faces_collected_this_video = 0
        faces_skipped_this_video = 0
        tracker = FaceTracker(frame_window=TRACKING_FRAME_WINDOW)

        video_start_time = time.time()
        face_detected_at_least_once = False

        print(f"   🔄 Extracting faces...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % BASE_FRAME_SKIP != 0:
                    continue

                try:
                    detections = RetinaFace.detect_faces(frame, threshold=0.5)

                    if not isinstance(detections, dict):
                        continue

                    if len(detections) > 0:
                        face_detected_at_least_once = True

                    for face_id, face_data in detections.items():
                        x1, y1, x2, y2 = face_data["facial_area"]
                        face_w, face_h = x2 - x1, y2 - y1

                        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                            faces_skipped_this_video += 1
                            continue

                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        box = (x1, y1, x2, y2)

                        if tracker.is_duplicate(frame_count, box):
                            faces_skipped_this_video += 1
                            continue

                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue

                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                        quality_ok, _, confidence = check_face_quality(face_rgb)
                        if not quality_ok:
                            faces_skipped_this_video += 1
                            continue

                        unique_id = str(uuid.uuid4())
                        tracker.add_face(frame_count, box, unique_id)

                        metadata = {
                            "id": unique_id,
                            "frame": frame_count,
                            "timestamp": float(frame_count / fps),
                            "video": os.path.basename(video_path),
                            "video_full_path": video_path,
                            "quality_confidence": float(confidence)
                        }

                        batch_encoder.add_face(face_rgb, metadata)

                        encoded = batch_encoder.process_batch(force=False)
                        for emb, meta in encoded:
                            faces_collected_this_video += 1
                            total_faces_collected += 1
                            batch_vectors.append((meta["id"], emb.tolist(), meta))

                            if len(batch_vectors) >= BATCH_SIZE:
                                index.upsert(vectors=batch_vectors, namespace=BULK_NAMESPACE)
                                batch_vectors = []

                        if total_faces_collected >= MAX_FACES_TO_COLLECT:
                            print(f"\n      ⚠️  Reached global max face limit ({MAX_FACES_TO_COLLECT})")
                            break

                except Exception as e:
                    print(f"      ⚠️  Frame {frame_count} error: {e}")
                    continue

                if total_faces_collected >= MAX_FACES_TO_COLLECT:
                    break

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"   ❌ ERROR: {error_msg}")
            cap.release()
            failed_videos.append({"video": video_path, "error": error_msg})
            bulk_results[video_path] = {"status": "error", "error": error_msg}
            continue

        cap.release()

        video_time = time.time() - video_start_time
        total_faces_skipped += faces_skipped_this_video

        if not face_detected_at_least_once:
            error_msg = f"No faces detected in entire video"
            print(f"   ⚠️  WARNING: {error_msg}")
            failed_videos.append({"video": video_path, "error": error_msg})
            bulk_results[video_path] = {"status": "warning", "faces": 0, "time": video_time, "error": error_msg}
        else:
            bulk_results[video_path] = {
                "status": "success",
                "faces": faces_collected_this_video,
                "skipped": faces_skipped_this_video,
                "time": video_time
            }
            print(f"\n   ✅ Extracted {faces_collected_this_video} faces in {video_time:.1f}s")

        if total_faces_collected >= MAX_FACES_TO_COLLECT:
            print(f"\n⚠️  Reached global limit - stopping batch processing")
            break

    # Flush remaining faces
    print(f"\n🔄 Uploading final batch...")
    encoded = batch_encoder.flush()
    for emb, meta in encoded:
        total_faces_collected += 1
        batch_vectors.append((meta["id"], emb.tolist(), meta))

    if batch_vectors:
        index.upsert(vectors=batch_vectors, namespace=BULK_NAMESPACE)
        print(f"   ✅ Uploaded {len(batch_vectors)} final faces")

    total_time = time.time() - total_start_time

    print(f"📊 BULK STORAGE SUMMARY")

    success_count = sum(1 for r in bulk_results.values() if r["status"] == "success")

    print(f"{'Video':<35} | {'Status':<15} | {'Faces':<10}")
    print(f"{'-'*35}-+-{'-'*15}-+-{'-'*10}")

    for video_path, result in bulk_results.items():
        video_name = os.path.basename(video_path)[:33]
        status = result["status"]

        if status == "success":
            print(f"{video_name:<35} | {'✅ Success':<15} | {result['faces']:<10}")
        elif status == "warning":
            print(f"{video_name:<35} | {'⚠️  No Faces':<15} | {result['faces']:<10}")
        else:
            print(f"{video_name:<35} | {'❌ Failed':<15} | {'-':<10}")

    print(f"\n{'='*70}")
    print(f"✅ Successful: {success_count}/{len(VIDEO_PATHS)} videos")
    print(f"📊 Total faces stored: {total_faces_collected:,}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"💾 Namespace: {BULK_NAMESPACE}")

    if failed_videos:
        print(f"\n{'='*70}")
        print(f"❌ FAILED VIDEOS ({len(failed_videos)}):")
        print(f"{'='*70}")
        for idx, failure in enumerate(failed_videos, 1):
            print(f"{idx}. {os.path.basename(failure['video'])}")
            print(f"   Error: {failure['error']}")
        print(f"{'='*70}")
    else:
        print(f"\n✅ All videos processed successfully!")

    print(f"{'='*70}")
