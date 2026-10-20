
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
DB_DIR = "./chroma_db"
COLLECTION_NAME = "heat_light_podcast"

# 1. Load the same embedding model used to build the database
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 2. Connect to the existing Chroma vector database
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

print("Stored records:", vector_store._collection.count())

# 3. Ask a question
question = input("\nAsk a question about the podcast: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# 4. Search for the four most relevant transcript chunks
print("\nSearching the vector database...")

results = vector_store.similarity_search_with_score(
    query=question,
    k=4,
)

# 5. Display the retrieved passages
print("\n========== RETRIEVED PASSAGES ==========")

if not results:
    print("No relevant passages found.")
else:
    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print("Episode ID:", document.metadata.get("video_id", "Unknown"))
        print("Chunk index:", document.metadata.get("chunk_index", "Unknown"))
        print("Distance:", round(float(distance), 4))
        print("Source:", document.metadata.get("source_url", "Unknown"))
        print("\nTranscript passage:")
        print(document.page_content[:700])
        print("-" * 60)

print("\nRetrieval test completed!")
