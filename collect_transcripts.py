
import json
from pathlib import Path

from langchain_community.document_loaders import YoutubeLoader

DATA_DIR = Path("data/transcripts")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_IDS = [
    "jdype8OYRrA",
    "H4Kt75nmgco",
    "nDP3w6Bgspk",
    "6RV0GEol_YY",
    "MreogGz9enc",
    "YD-yl9FSSlo",
    "CfAswwELGkg",
    "JZsRnpHG-iU",
    "6G5idtDdQ2Q",
    "Gyh2Lk5iPWY",
]


def collect_transcript(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    text_path = DATA_DIR / f"{video_id}.txt"
    metadata_path = DATA_DIR / f"{video_id}.json"

    # Reuse an existing transcript instead of downloading it again.
    if text_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if text_path.stat().st_size > 0 and metadata.get("video_id") == video_id:
                print(f"SKIP: Episode {video_id} already saved")
                return True
        except (json.JSONDecodeError, OSError):
            pass

    try:
        print(f"\nCollecting: {video_id}")

        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["en"],
        )

        documents = loader.load()

        if not documents:
            print(f"FAILED: No transcript returned for {video_id}")
            return False

        transcript = "\n".join(
            doc.page_content for doc in documents
        ).strip()

        if not transcript:
            print(f"FAILED: Empty transcript for {video_id}")
            return False

        metadata = {
            "video_id": video_id,
            "url": url,
            "language": "en",
            "transcript_file": text_path.name,
            "character_count": len(transcript),
        }

        text_path.write_text(transcript, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print(f"SAVED: {video_id} ({len(transcript)} characters)")
        return True

    except Exception as error:
        print(f"FAILED: {video_id}")
        print(repr(error))
        return False


if __name__ == "__main__":
    successful = 0
    failed = []

    for video_id in VIDEO_IDS:
        if collect_transcript(video_id):
            successful += 1
        else:
            failed.append(video_id)

    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Successful episodes: {successful}/{len(VIDEO_IDS)}")
    print(f"Failed episodes: {len(failed)}")

    if failed:
        print("Retry these video IDs:")
        for video_id in failed:
            print(video_id)
