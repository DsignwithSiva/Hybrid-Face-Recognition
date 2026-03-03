import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import MODE, DEVICE, VIDEO_NAMESPACE

print(f"⚙️  HYBRID FACE SEARCH SYSTEM")
print(f"   - Mode: {MODE.upper()}")
print(f"   - Video Namespace: {VIDEO_NAMESPACE}")

# Imports are deferred here so the env vars above are set before TF loads
from store_modes import (
    store_all_faces_from_video,
    bulk_store_multiple_videos,
    bulk_store_multiple_videos_single_namespace,
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
    print("\n🚀 HYBRID FACE RECOGNITION SYSTEM")
    print("="*70)

    if MODE == "store":
        store_all_faces_from_video()
        print("\n💡 Next steps:")
        print("   1. Change MODE to 'search', 'batch_search', 'multi_video_search', or 'ultimate_search'")
        print("   2. Set appropriate IMAGE_PATH / BATCH_IMAGE_PATHS / VIDEO_PATHS")
        print("   3. Run the script again - search will be INSTANT!")

    elif MODE == "search":
        search_for_person_in_stored_faces()
        print("\n💡 Want to search for another person?")
        print("   - Change IMAGE_PATH and run again")
        print("   - Or try batch_search for multiple people!")

    elif MODE == "batch_search":
        batch_search_multiple_people()
        print("\n💡 Searched multiple people in one video!")
        print("   - To search across multiple videos, try 'ultimate_search' mode")

    elif MODE == "multi_video_search":
        multi_video_search_one_person()
        print("\n💡 Searched one person across multiple videos!")
        print("   - To search multiple people, try 'ultimate_search' mode")

    elif MODE == "ultimate_search":
        ultimate_search()
        print("\n💡 Ultimate search complete!")
        print("   - Searched all people across all videos")

    elif MODE == "bulk_store":
        bulk_store_multiple_videos()
        print("\n💡 Bulk storage complete!")
        print("   - All videos now searchable")
        print("   - Try 'ultimate_search' to find people across all videos")

    else:
        print(f"❌ ERROR: Invalid MODE '{MODE}'")
        print("   Valid options: 'store', 'search', 'batch_search', 'multi_video_search', 'ultimate_search', 'bulk_store'")

    print("\n✨ Complete!")
