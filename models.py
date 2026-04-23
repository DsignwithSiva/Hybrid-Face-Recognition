from facenet_pytorch import InceptionResnetV1
from pinecone import Pinecone
from config import DEVICE, PINECONE_API_KEY, PINECONE_INDEX_NAME

# -------------------------------
# LOAD FACENET MODEL
# -------------------------------

print("🔧 Loading FaceNet model...")

try:
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    print(f"✅ FaceNet loaded on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading FaceNet: {e}")
    raise

# -------------------------------
# INITIALIZE PINECONE
# -------------------------------

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("✅ Pinecone connected successfully")
except Exception as e:
    print(f"❌ Pinecone connection error: {e}")
    raise