from facenet_pytorch import InceptionResnetV1
from pinecone import Pinecone

from config import DEVICE, PINECONE_API_KEY, PINECONE_INDEX_NAME

# -------------------------------
# INITIALIZE MODELS
# -------------------------------

print("🔧 Loading FaceNet model...")
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
