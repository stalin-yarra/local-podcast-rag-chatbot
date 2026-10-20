
# Local Podcast RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers
questions about YouTube podcast transcripts using local embeddings,
Chroma vector search, and a locally running language model.

## Features

- Collect transcripts from YouTube videos
- Split transcripts into overlapping text chunks
- Generate embeddings locally with Sentence Transformers
- Store and retrieve chunks using ChromaDB
- Generate answers using Qwen 2.5 3B through Ollama
- Include transcript passage citations and YouTube source URLs
- Handle questions that are not supported by retrieved context

## Tech Stack

- Python
- LangChain
- Sentence Transformers
- ChromaDB
- Ollama
- Qwen 2.5 3B

## Architecture

1. Collect YouTube transcripts.
2. Split transcripts into chunks.
3. Generate local embeddings for the chunks.
4. Store the embeddings and metadata in ChromaDB.
5. Embed a user's question and retrieve relevant chunks.
6. Pass the question and retrieved context to the local LLM.
7. Generate an answer with passage citations and source URLs.

## Prerequisites

- Python
- Git
- Ollama

## Setup

Clone this repository and enter the project directory:

```bash
git clone YOUR_REPOSITORY_URL
cd Rag_app
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the language model:

```powershell
ollama pull qwen2.5:3b
```

## Prepare the Data

Run the transcript collection script:

```powershell
python .\collect_transcripts.py
```

Then generate transcript chunks:

```powershell
python .\chunk_transcripts.py
```

Build the Chroma index:

```powershell
python .\index_chroma.py
```

## Run the Chatbot

Ensure Ollama is running, then execute:

```powershell
python .\rag_chat.py
```

Ask questions about the collected transcripts in the terminal.
Type `exit` to stop.

## Notes

- Embeddings and answer generation run locally.
- The first embedding-model download requires an internet connection.
- Ollama must have the selected language model installed.
- The Chroma database and transcript data are generated locally.
- Answers depend on the quality of retrieval and the local language model.
