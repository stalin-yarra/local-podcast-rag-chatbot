
import json
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHUNKS_FILE = Path("data/transcripts/chunks.jsonl")
DB_DIR = "./chroma_db"
COLLECTION_NAME = "heat_light_podcast"

# 1. Load transcript chunks
if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"Missing file: {CHUNKS_FILE.resolve()}")

documents = []
ids = []

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        record = json.loads(line)

        documents.append(
            Document(
                page_content=record["text"],
                metadata={
                    "video_id": record["video_id"],
                    "source_url": record["source_url"],
                    "chunk_index": record["chunk_index"],
                },
            )
        )
        ids.append(record["chunk_id"])

if not documents:
    raise RuntimeError("No transcript chunks found.")

if len(ids) != len(set(ids)):
    raise ValueError("Duplicate chunk IDs found.")

print("Loaded chunks:", len(documents))

# 2. Load local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 3. Recreate the target collection
client = chromadb.PersistentClient(path=DB_DIR)

try:
    client.delete_collection(name=COLLECTION_NAME)
    print("Deleted previous collection.")
except Exception:
    print("No previous collection to delete.")

client.create_collection(name=COLLECTION_NAME)

# 4. Connect LangChain to the new collection
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# 5. Add chunks in batches
batch_size = 50

for start in range(0, len(documents), batch_size):
    end = min(start + batch_size, len(documents))

    vector_store.add_documents(
        documents=documents[start:end],
        ids=ids[start:end],
    )

    print(f"Indexed {end}/{len(documents)} chunks")

# 6. Verify the saved records using a fresh client
check_client = chromadb.PersistentClient(path=DB_DIR)
saved_collection = check_client.get_collection(COLLECTION_NAME)

print("\n========== INDEXING VERIFICATION ==========")
print("Chunks loaded:", len(documents))
print("Records in collection:", saved_collection.count())
print("Database:", Path(DB_DIR).resolve())
print("Collection:", COLLECTION_NAME)
print("Embedding model: sentence-transformers/all-MiniLM-L6-v2")

if saved_collection.count() != len(documents):
    raise RuntimeError("Index verification failed: record counts do not match.")

print("\nSUCCESS: All transcript chunks are indexed.")
