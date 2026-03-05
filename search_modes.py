import os
import cv2
import torch
import numpy as np

# Fix broken TensorFlow __version__ on Windows
try:
    import tensorflow as _tf
    if not hasattr(_tf, '__version__'):
        _tf.__version__ = '2.15.0'
except Exception:
    pass

from retinaface import RetinaFace

from config import (
    IMAGE_PATH, VIDEO_PATH, VIDEO_PATHS, VIDEO_NAMESPACE,
    BATCH_IMAGE_PATHS, DIST_THRESHOLD, TEMPORAL_CLUSTER_THRESHOLD,
    TOP_K_RESULTS, DEVICE
)
from utils import l2_normalize, TemporalClusterer
from models import model, index

# Original filenames for batch search (set by server before calling batch_search_multiple_people)
BATCH_IMAGE_NAMES: list = []


# ===============================
# HELPER FUNCTION
# ===============================

def encode_reference_image(image_path: str) -> np.ndarray:
    """Load and encode a reference image into embedding."""
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_faces = RetinaFace.detect_faces(ref_img_rgb, threshold=0.5)

    if not isinstance(ref_faces, dict) or len(ref_faces) == 0:
        raise ValueError(f"No faces detected in: {image_path}")

    if len(ref_faces) > 1:
        largest_face_key = max(ref_faces.keys(),
                               key=lambda k: (ref_faces[k]['facial_area'][2] - ref_faces[k]['facial_area'][0]) *
                                             (ref_faces[k]['facial_area'][3] - ref_faces[k]['facial_area'][1]))
        ref_face_data = ref_faces[largest_face_key]
    else:
        ref_face_data = list(ref_faces.values())[0]

    x1, y1, x2, y2 = ref_face_data["facial_area"]
    ref_face = ref_img_rgb[y1:y2, x1:x2]
    ref_face = cv2.resize(ref_face, (160, 160))

    ref_tensor = (
        torch.tensor(ref_face)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0) / 255.0
    ).to(DEVICE)

    with torch.no_grad():
        ref_emb = model(ref_tensor).cpu().numpy()[0]

    return l2_normalize(ref_emb)


# ===============================
# MODE 2: SEARCH FOR PERSON
# ===============================

def search_for_person_in_stored_faces():
    """
    Search for a specific person in previously stored faces.
    This is INSTANT - no video processing needed!
    """
    try:
        stats = index.describe_index_stats()
        namespace_count = stats.get('namespaces', {}).get(VIDEO_NAMESPACE, {}).get('vector_count', 0)

        if namespace_count == 0:
            print(f"\n❌ ERROR: No faces found in namespace '{VIDEO_NAMESPACE}'")
            print(f"   Please run in 'store' mode first to process the video.")
            return

    except Exception as e:
        print(f"\n❌ ERROR: Could not access namespace: {e}")
        return

    ref_img = cv2.imread(IMAGE_PATH)
    if ref_img is None:
        raise FileNotFoundError(f"Could not load reference image: {IMAGE_PATH}")

    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_faces = RetinaFace.detect_faces(ref_img_rgb, threshold=0.5)

    if not isinstance(ref_faces, dict) or len(ref_faces) == 0:
        raise ValueError(f"No faces detected in reference image: {IMAGE_PATH}")

    if len(ref_faces) > 1:
        print(f"⚠️  Found {len(ref_faces)} faces. Using largest face.")
        largest_face_key = max(ref_faces.keys(),
                               key=lambda k: (ref_faces[k]['facial_area'][2] - ref_faces[k]['facial_area'][0]) *
                                             (ref_faces[k]['facial_area'][3] - ref_faces[k]['facial_area'][1]))
        ref_face_data = ref_faces[largest_face_key]
    else:
        ref_face_data = list(ref_faces.values())[0]

    x1, y1, x2, y2 = ref_face_data["facial_area"]
    ref_face = ref_img_rgb[y1:y2, x1:x2]
    ref_face = cv2.resize(ref_face, (160, 160))

    ref_tensor = (
        torch.tensor(ref_face)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0) / 255.0
    ).to(DEVICE)

    with torch.no_grad():
        ref_emb = model(ref_tensor).cpu().numpy()[0]
    ref_emb = l2_normalize(ref_emb)

    import time
    search_start = time.time()

    results = index.query(
        vector=ref_emb.tolist(),
        top_k=TOP_K_RESULTS,
        include_metadata=True,
        namespace=VIDEO_NAMESPACE
    )

    search_time = time.time() - search_start
    print(f"✅ Search completed in {search_time:.2f}s")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    if fps == 0:
        print("⚠️  Warning: Could not read FPS from video. Defaulting to 25 FPS.")
        fps = 25.0

    all_matches = []
    clusterer = TemporalClusterer(frame_threshold=TEMPORAL_CLUSTER_THRESHOLD)

    for match in results['matches']:
        distance = 1 - match['score']
        if distance < DIST_THRESHOLD:
            confidence = match['metadata'].get('quality_confidence', 0.5)
            frame_num = match['metadata']['frame']

            clusterer.add_detection(frame_num, distance, confidence)
            all_matches.append({
                'frame': frame_num,
                'distance': distance,
                'similarity': match['score'],
                'confidence': confidence,
                'metadata': match['metadata']
            })

    clusters = clusterer.get_clusters()

    print(f"\n📊 Search Results:")
    print(f"   - Matching frames: {len(all_matches)}")
    print(f"   - Appearance segments: {len(clusters)}")


    if len(clusters) > 0:
        print("✅ RESULT: Person IS PRESENT in the video")
        print(f"\n🎯 Found {len(clusters)} appearance segment(s):\n")

        for i, cluster in enumerate(clusters, 1):
            start_time_sec = cluster['start_frame'] / fps
            end_time_sec = cluster['end_frame'] / fps
            duration_sec = end_time_sec - start_time_sec

            confidence_pct = cluster['avg_confidence'] * 100
            match_quality = "STRONG" if cluster['best_distance'] < 0.30 else "GOOD"

            print(f"   Segment {i}:")
            print(f"      📍 Time: {start_time_sec:.2f}s - {end_time_sec:.2f}s (duration: {duration_sec:.2f}s)")
            print(f"      🎞️  Frames: {cluster['start_frame']} - {cluster['end_frame']}")
            print(f"      ⭐ Confidence: {confidence_pct:.1f}%")
            print(f"      💪 Quality: {match_quality}")
            print()

        best_cluster = min(clusters, key=lambda c: c['best_distance'])
        print(f"🏆 Best Match Summary:")
        print(f"   - Timestamp: {best_cluster['start_frame'] / fps:.2f}s")
        print(f"   - Frame: {best_cluster['start_frame']}")
    else:
        print("❌ RESULT: Person NOT PRESENT in the video")

        if all_matches:
            closest = min(all_matches, key=lambda x: x['distance'])
            print(f"\n   💡 Closest match: frame {closest['frame']} (distance {closest['distance']:.4f})")

# ===============================
# MODE 3: BATCH SEARCH (Multiple People, One Video)
# ===============================

def batch_search_multiple_people():
    """
    Search for MULTIPLE people in ONE namespace (could be multiple videos).
    Now supports searching across bulk-stored videos.
    """

    try:
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})

        if not namespaces:
            print(f"\n❌ ERROR: No namespaces found!")
            print(f"   Run 'bulk_store' mode first to process videos.")
            return

        ns_list = list(namespaces.keys())

        # When called from the server VIDEO_NAMESPACE is pre-set — skip the prompt.
        if VIDEO_NAMESPACE:
            search_namespace   = VIDEO_NAMESPACE
            namespace_count    = namespaces.get(search_namespace, {}).get('vector_count', 0)
            if namespace_count == 0:
                print(f"\n❌ ERROR: Namespace '{search_namespace}' not found or is empty!")
                print(f"   Available namespaces: {', '.join(ns_list)}")
                return
            print(f"\n📁 Namespace  : {search_namespace} ({namespace_count} faces indexed)")
        else:
            # Interactive CLI fallback
            print(f"\nAvailable namespaces:")
            for idx, ns_name in enumerate(ns_list, 1):
                count = namespaces[ns_name].get('vector_count', 0)
                print(f"   {idx}. {ns_name} ({count} faces)")

            selection = input(f"\nSelect namespace (1-{len(ns_list)}) or press Enter for bulk: ").strip()

            if selection == "":
                search_namespace = "bulk_videos_combined"
            else:
                try:
                    search_namespace = ns_list[int(selection) - 1]
                except Exception:
                    print("Invalid selection")
                    return

            namespace_count = namespaces.get(search_namespace, {}).get('vector_count', 0)

            if namespace_count == 0:
                print(f"\n❌ ERROR: Namespace '{search_namespace}' is empty!")
                return

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return

    print(f"\n👥 Searching for {len(BATCH_IMAGE_PATHS)} people...")

    all_results = {}
    failed_images = []

    for idx, image_path in enumerate(BATCH_IMAGE_PATHS, 1):
        # Use original upload name when available; fall back to temp basename
        display_name = (
            BATCH_IMAGE_NAMES[idx - 1]
            if BATCH_IMAGE_NAMES and idx - 1 < len(BATCH_IMAGE_NAMES)
            else os.path.basename(image_path)
        )

        print(f"\n{'─'*55}")
        print(f"[{idx}/{len(BATCH_IMAGE_PATHS)}] 👤 Person: {display_name}")
        print(f"{'─'*55}")

        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            print(f"   ❌ ERROR: {error_msg}")
            failed_images.append({"image": image_path, "error": error_msg, "name": display_name})
            all_results[image_path] = {"status": "error", "error": error_msg, "name": display_name}
            continue

        print(f"   🖼️  Encoding face from image...")
        try:
            ref_emb = encode_reference_image(image_path)
            print(f"   ✔  Face encoded successfully")
        except Exception as e:
            error_msg = f"Could not detect face in image: {str(e)}"
            print(f"   ❌ ERROR: {error_msg}")
            failed_images.append({"image": image_path, "error": error_msg, "name": display_name})
            all_results[image_path] = {"status": "error", "error": error_msg, "name": display_name}
            continue

        try:
            results = index.query(
                vector=ref_emb.tolist(),
                top_k=TOP_K_RESULTS,
                include_metadata=True,
                namespace=search_namespace
            )
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            print(f"   ❌ ERROR: {error_msg}")
            failed_images.append({"image": image_path, "error": error_msg, "name": display_name})
            all_results[image_path] = {"status": "error", "error": error_msg, "name": display_name}
            continue

        matches = []
        all_distances = []
        clusterer = TemporalClusterer(frame_threshold=TEMPORAL_CLUSTER_THRESHOLD)

        for match in results['matches']:
            distance = 1 - match['score']
            all_distances.append(distance)
            if distance < DIST_THRESHOLD:
                confidence = match['metadata'].get('quality_confidence', 0.5)
                frame_num = match['metadata']['frame']
                clusterer.add_detection(frame_num, distance, confidence)
                matches.append({
                    'frame': frame_num,
                    'distance': distance,
                    'video': match['metadata'].get('video', 'unknown')
                })

        clusters = clusterer.get_clusters()

        all_results[image_path] = {
            "status": "found" if len(clusters) > 0 else "not_found",
            "matches": len(matches),
            "segments": clusters,
            "name": display_name
        }

        if len(clusters) > 0:
            best_dist = min(m['distance'] for m in matches)
            videos_found = set(m['video'] for m in matches)
            print(f"   ✅ RESULT  : FOUND in video")
            print(f"   📊 Matches : {len(matches)} frame(s) across {len(clusters)} segment(s)")
            print(f"   🎬 Video(s) : {', '.join(videos_found)}")
            for seg_i, seg in enumerate(clusters, 1):
                fps_est = 25.0
                t_start = seg['start_frame'] / fps_est
                t_end   = seg['end_frame']   / fps_est
                print(f"      Segment {seg_i}: frames {seg['start_frame']}–{seg['end_frame']}  ({t_start:.1f}s – {t_end:.1f}s)")
        else:
            closest_dist = min(all_distances) if all_distances else None
            print(f"   ❌ RESULT  : NOT FOUND in video")
            if closest_dist is not None:
                print(f"   💡 Closest match distance: {closest_dist:.4f}  (threshold: {DIST_THRESHOLD})")
                if closest_dist < DIST_THRESHOLD + 0.10:
                    print(f"   ⚠️  Very close! Try lowering threshold below {closest_dist:.2f} to include this match.")
            else:
                print(f"   ℹ️  No candidates returned from vector DB.")


    found_count = sum(1 for r in all_results.values() if r["status"] == "found")

    print(f"\n{'═'*55}")
    print(f"📊 BATCH SEARCH SUMMARY  ({found_count}/{len(BATCH_IMAGE_PATHS)} people found)")
    print(f"{'═'*55}")

    for i, (image_path, result) in enumerate(all_results.items(), 1):
        name = result.get("name") or os.path.basename(image_path)
        if result["status"] == "found":
            print(f"  {i}. ✅ FOUND    — {name}")
            print(f"        {result['matches']} frame match(es) in {len(result['segments'])} segment(s)")
        elif result["status"] == "not_found":
            print(f"  {i}. ❌ NOT FOUND — {name}")
        else:
            print(f"  {i}. ⚠️  ERROR     — {name}")
            print(f"        {result.get('error', 'Unknown error')}")

    if failed_images:
        print(f"\n  ⚠️  {len(failed_images)} image(s) could not be processed (see errors above)")

    print(f"{'═'*55}")
    print(f"\n✅ Final Result: {found_count}/{len(BATCH_IMAGE_PATHS)} people found in '{search_namespace}'")


# ===============================
# MODE 4: MULTI-VIDEO SEARCH (One Person, Multiple Videos)
# ===============================

def multi_video_search_one_person():
    """
    Search for ONE person across MULTIPLE videos.
    Useful for: "Find this actor in all my movies"
    """

    try:
        ref_emb = encode_reference_image(IMAGE_PATH)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return


    all_results = {}

    for idx, video_path in enumerate(VIDEO_PATHS, 1):

        # If it has a file extension (.mp4 etc.) derive namespace from filename;
        # otherwise the value IS the namespace (sent directly from the server UI)
        if os.path.splitext(video_path)[1]:
            video_namespace = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"
        else:
            video_namespace = video_path

        try:
            stats = index.describe_index_stats()
            namespace_count = stats.get('namespaces', {}).get(video_namespace, {}).get('vector_count', 0)

            if namespace_count == 0:
                print(f"   ⚠️  Namespace '{video_namespace}' not stored - skipping")
                all_results[video_path] = {"status": "not_stored"}
                continue

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results[video_path] = {"status": "error", "message": str(e)}
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if fps == 0:
            fps = 25.0

        results = index.query(
            vector=ref_emb.tolist(),
            top_k=TOP_K_RESULTS,
            include_metadata=True,
            namespace=video_namespace
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
                    'distance': distance,
                    'confidence': confidence
                })

        clusters = clusterer.get_clusters()

        all_results[video_path] = {
            "status": "found" if len(clusters) > 0 else "not_found",
            "matches": len(matches),
            "segments": clusters,
            "fps": fps
        }

        if len(clusters) > 0:
            print(f"   ✅ FOUND - {len(clusters)} segment(s)")
            for i, cluster in enumerate(clusters, 1):
                start_sec = cluster['start_frame'] / fps
                end_sec = cluster['end_frame'] / fps
                print(f"      Segment {i}: {start_sec:.1f}s - {end_sec:.1f}s")
        else:
            print(f"   ❌ NOT FOUND")


    print(f"📊 MULTI-VIDEO SEARCH SUMMARY")

    print(f"Person: {IMAGE_PATH}\n")

    found_count = sum(1 for r in all_results.values() if r["status"] == "found")

    for video_path, result in all_results.items():
        status_icon = "✅" if result["status"] == "found" else "❌"
        video_name = os.path.basename(video_path)

        print(f"{status_icon} {video_name}")

        if result["status"] == "found":
            for i, cluster in enumerate(result["segments"], 1):
                start_sec = cluster['start_frame'] / result['fps']
                end_sec = cluster['end_frame'] / result['fps']
                best_dist = cluster['best_distance']
                print(f"   └─ Segment {i}: {start_sec:.1f}s - {end_sec:.1f}s (distance: {best_dist:.3f})")
        elif result["status"] == "not_found":
            print(f"   └─ No matches found")
        elif result["status"] == "not_stored":
            print(f"   └─ Video not in database (run store mode first)")
        else:
            print(f"   └─ Error: {result.get('message', 'Unknown')}")
        print()

    print(f"{'='*70}")
    print(f"✅ Found in: {found_count}/{len(VIDEO_PATHS)} videos")
    print(f"❌ Not found in: {len(VIDEO_PATHS) - found_count}/{len(VIDEO_PATHS)} videos")
    print(f"{'='*70}")


# ===============================
# MODE 5: ULTIMATE SEARCH (Multiple People × Multiple Videos)
# ===============================

def ultimate_search():
    """
    Search for MULTIPLE people across MULTIPLE videos.
    The ultimate search mode!
    """
    print("\n" + "="*70)
    print("🚀 MODE: ULTIMATE SEARCH - Multiple People × Multiple Videos")
    print("="*70)

    print(f"\n📊 Search Configuration:")
    print(f"   - People: {len(BATCH_IMAGE_PATHS)}")
    print(f"   - Videos: {len(VIDEO_PATHS)}")
    print(f"   - Total searches: {len(BATCH_IMAGE_PATHS) * len(VIDEO_PATHS)}")

    master_results = {}

    for person_idx, image_path in enumerate(BATCH_IMAGE_PATHS, 1):
        print(f"\n{'#'*70}")
        print(f"👤 PERSON [{person_idx}/{len(BATCH_IMAGE_PATHS)}]: {os.path.basename(image_path)}")
        print(f"{'#'*70}")

        try:
            ref_emb = encode_reference_image(image_path)
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            master_results[image_path] = {"error": str(e)}
            continue

        person_results = {}

        for video_idx, video_path in enumerate(VIDEO_PATHS, 1):
            # If it has a file extension (.mp4 etc.) derive namespace from filename;
            # otherwise the value IS the namespace (sent directly from the server UI)
            if os.path.splitext(video_path)[1]:
                video_namespace = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"
            else:
                video_namespace = video_path

            try:
                stats = index.describe_index_stats()
                namespace_count = stats.get('namespaces', {}).get(video_namespace, {}).get('vector_count', 0)

                if namespace_count == 0:
                    print(f"     ⚠️  Namespace '{video_namespace}' not stored - skipping")
                    person_results[video_path] = {"status": "not_stored"}
                    continue
            except Exception as e:
                print(f"     ❌ Error: {e}")
                person_results[video_path] = {"status": "error", "message": str(e)}
                continue

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if fps == 0:
                fps = 25.0

            results = index.query(
                vector=ref_emb.tolist(),
                top_k=TOP_K_RESULTS,
                include_metadata=True,
                namespace=video_namespace
            )

            matches = []
            clusterer = TemporalClusterer(frame_threshold=TEMPORAL_CLUSTER_THRESHOLD)

            for match in results['matches']:
                distance = 1 - match['score']
                if distance < DIST_THRESHOLD:
                    confidence = match['metadata'].get('quality_confidence', 0.5)
                    frame_num = match['metadata']['frame']
                    clusterer.add_detection(frame_num, distance, confidence)
                    matches.append({'frame': frame_num, 'distance': distance})

            clusters = clusterer.get_clusters()

            person_results[video_path] = {
                "status": "found" if len(clusters) > 0 else "not_found",
                "matches": len(matches),
                "segments": clusters,
                "fps": fps
            }

            if len(clusters) > 0:
                print(f"     ✅ FOUND - {len(clusters)} segment(s)")
            else:
                print(f"     ❌ NOT FOUND")

        master_results[image_path] = person_results

    print(f"🏆 ULTIMATE SEARCH RESULTS")

    for image_path, video_results in master_results.items():
        if "error" in video_results:
            print(f"{os.path.basename(image_path):<25} | {'ERROR':<15} | -")
            continue

        found_in = sum(1 for r in video_results.values() if r.get("status") == "found")
        total_segments = sum(len(r.get("segments", [])) for r in video_results.values())

    print(f"\n{'='*70}")
    print(f"📋 DETAILED RESULTS")
    print(f"{'='*70}\n")

    for image_path, video_results in master_results.items():
        print(f"👤 {os.path.basename(image_path)}")
        print(f"   {'-'*60}")

        if "error" in video_results:
            print(f"   ❌ Error: {video_results['error']}\n")
            continue

        for video_path, result in video_results.items():
            video_name = os.path.basename(video_path)
            status = result.get("status", "unknown")

            if status == "found":
                print(f"   ✅ {video_name}")
                for i, cluster in enumerate(result["segments"], 1):
                    start_sec = cluster['start_frame'] / result['fps']
                    end_sec = cluster['end_frame'] / result['fps']
                    print(f"      └─ Segment {i}: {start_sec:.1f}s - {end_sec:.1f}s")
            elif status == "not_found":
                print(f"   ❌ {video_name} - Not found")
            elif status == "not_stored":
                print(f"   ⚠️  {video_name} - Not in database")

        print()

    print(f"{'='*70}")
