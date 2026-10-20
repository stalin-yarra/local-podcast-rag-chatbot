
from pathlib import Path
import chromadb

DB_DIR = "./chroma_db"

print("Current folder:", Path.cwd())
print("Database path:", Path(DB_DIR).resolve())
print("Database folder exists:", Path(DB_DIR).exists())

client = chromadb.PersistentClient(path=DB_DIR)

print("\nCollections found:")
collections = client.list_collections()

if not collections:
    print("No collections found in this database.")
else:
    for collection in collections:
        print(
            f"Name: {collection.name}, "
            f"Records: {collection.count()}"
        )
