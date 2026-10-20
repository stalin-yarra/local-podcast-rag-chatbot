
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TRANSCRIPT_DIR = Path("data/transcripts")
OUTPUT_FILE = TRANSCRIPT_DIR / "chunks.jsonl"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def chunk_transcript(metadata_path):
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    transcript_path = (
        TRANSCRIPT_DIR / metadata["transcript_file"]
    )

    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}")
        return []

    transcript = transcript_path.read_text(encoding="utf-8")

    if not transcript.strip():
        print(f"Empty transcript: {transcript_path}")
        return []

    chunks = splitter.split_text(transcript)
    records = []

    for index, chunk in enumerate(chunks):
        records.append({
            "chunk_id": f"{metadata['video_id']}_{index:04d}",
            "video_id": metadata["video_id"],
            "source_url": metadata["url"],
            "chunk_index": index,
            "text": chunk,
        })

    print(f"\nEpisode: {metadata['video_id']}")
    print(f"Transcript characters: {len(transcript)}")
    print(f"Chunks created: {len(records)}")

    if records:
        print("\nFirst chunk preview:\n")
        print(records[0]["text"][:500])

    return records


def main():
    metadata_files = sorted(
        TRANSCRIPT_DIR.glob("*.json")
    )

    if not metadata_files:
        print("No transcript metadata files found.")
        return

    all_records = []

    for metadata_path in metadata_files:
        all_records.extend(
            chunk_transcript(metadata_path)
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in all_records:
            file.write(json.dumps(record) + "\n")

    print(f"\nTotal chunks: {len(all_records)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
