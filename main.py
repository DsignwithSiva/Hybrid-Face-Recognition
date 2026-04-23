import warnings
warnings.filterwarnings("ignore")

from config import MODE, VIDEO_NAMESPACE, VIDEO_PATH, IMAGE_PATH

print(f"🚀 Hybrid Face Recognition System")
print(f"   - Mode: {MODE.upper()}")

if MODE in ('store', 'multi_video_search', 'bulk_store'):
    print(f"   - Video path(s): {VIDEO_PATH}")

if MODE in ('search', 'batch_search', 'ultimate_search'):
    print(f"   - Image path(s): {IMAGE_PATH}")

# Import modules
from store_modes import (
    store_all_faces_from_video,
    bulk_store_multiple_videos,
)

from search_modes import (
    search_for_person_in_stored_faces,
    batch_search_multiple_people,
    multi_video_search_one_person,
    ultimate_search,
)

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":

    try:
        if MODE == "store":
            store_all_faces_from_video()

        elif MODE == "search":
            search_for_person_in_stored_faces()

        elif MODE == "batch_search":
            batch_search_multiple_people()
            print("\n💡 Searched multiple people in one video!")

        elif MODE == "multi_video_search":
            multi_video_search_one_person()
            print("\n💡 Searched one person across multiple videos!")

        elif MODE == "ultimate_search":
            ultimate_search()
            print("\n💡 Ultimate search complete!")

        elif MODE == "bulk_store":
            bulk_store_multiple_videos()
            print("\n💡 Bulk storage complete!")

        else:
            print(f"❌ ERROR: Invalid MODE '{MODE}'")

    except Exception as e:
        print(f"❌ Runtime Error: {e}")

    print("\n✨ Process Complete!")