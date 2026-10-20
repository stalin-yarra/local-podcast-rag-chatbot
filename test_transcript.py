
from langchain_community.document_loaders import YoutubeLoader

youtube_url = "https://www.youtube.com/watch?v=jdype8OYRrA&list=PLWprQDlM_bY-5FVBnqLZ3-vGnInmrs4LE"

try:
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=False,
        language=["en"],
    )

    documents = loader.load()

    if not documents:
        print("No transcript documents were returned.")
    else:
        transcript = "\n".join(
            document.page_content for document in documents
        )

        print("Transcript extraction successful!")
        print("Number of documents:", len(documents))
        print("Transcript length:", len(transcript), "characters")
        print("\nFirst 1500 characters:\n")
        print(transcript[:1500])

        print("\nMetadata:")
        for document in documents:
            print(document.metadata)

except Exception as error:
    print("Transcript extraction failed.")
    print("Error:", repr(error))
