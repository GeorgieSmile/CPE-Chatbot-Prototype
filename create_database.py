#!/usr/bin/env python3
"""
Create (or refresh) a Chroma vector DB from Markdown FAQs.

- Keeps chunking simple with MarkdownHeaderTextSplitter
- Stores useful metadata (title/section/question + source filename)
- Uses OpenAI text-embedding-3-small (cheap + good)
- Adds --reset flag to rebuild the DB
- Sensible CLI flags for paths and collection name
"""

import argparse
import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# ----------------------------
# Defaults (override via CLI)
# ----------------------------
DEFAULT_CHROMA_PATH = "chroma"
DEFAULT_DATA_PATH = "data"           # directory containing *.md
DEFAULT_COLLECTION = "siit-faqs"


def require_openai_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... "
            "or export it in your shell."
        )


def load_markdown_documents(data_path: str) -> List:
    """
    Load all .md files under data_path (recursively).
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    # DirectoryLoader will attach 'source' metadata with the file path
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No Markdown files found in: {data_path}")

    return documents


def split_text(documents: List) -> List:
    """
    Split Markdown by headers and keep helpful metadata.
    headers_to_split_on define the hierarchy that becomes metadata.
    """
    headers_to_split_on = [
        ("#", "title"),
        ("##", "section"),
        ("###", "question"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    chunks = []
    for doc in documents:
        parts = splitter.split_text(doc.page_content)

        # Enrich metadata on each chunk
        src = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
        src_name = os.path.basename(src)

        for p in parts:
            # carry forward file source
            p.metadata["source"] = src_name or src

            # make sure keys exist even if header missing
            p.metadata.setdefault("title", p.metadata.get("title", ""))
            p.metadata.setdefault("section", p.metadata.get("section", ""))
            p.metadata.setdefault("question", p.metadata.get("question", ""))

            # tiny cleanup
            p.page_content = " ".join(p.page_content.split())

        chunks.extend(parts)

    if not chunks:
        raise RuntimeError("No chunks produced. Check your Markdown formatting.")
    return chunks


def build_chroma(chunks: List, chroma_path: str, collection_name: str):
    """
    Build and persist a Chroma DB from chunks.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
        collection_name=collection_name,
    )
    # Chroma 0.4+ auto-persists; no db.persist() needed
    return db


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Chroma DB from Markdown FAQs.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Folder containing .md files")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="Chroma persistence folder")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--reset", action="store_true", help="Delete the existing Chroma folder before building")
    return parser.parse_args()


def main():
    args = parse_args()
    require_openai_key()

    print(f"[1/4] Loading markdown from: {args.data_path}")
    docs = load_markdown_documents(args.data_path)
    print(f"       Loaded {len(docs)} files")

    print("[2/4] Splitting into header-aware chunks…")
    chunks = split_text(docs)
    print(f"       Created {len(chunks)} chunks")

    if args.reset and os.path.exists(args.chroma_path):
        print(f"[3/4] --reset specified: removing {args.chroma_path}")
        shutil.rmtree(args.chroma_path)

    print(f"[4/4] Building Chroma DB at: {args.chroma_path} (collection: {args.collection})")
    build_chroma(chunks, args.chroma_path, args.collection)

    # Small report
    titles = sum(1 for c in chunks if c.metadata.get("title"))
    sections = sum(1 for c in chunks if c.metadata.get("section"))
    questions = sum(1 for c in chunks if c.metadata.get("question"))

    print("✅ Done.")
    print(f"   Chunks: {len(chunks)} | Titles: {titles} | Sections: {sections} | Questions: {questions}")
    print(f"   Persisted to: {args.chroma_path}")


if __name__ == "__main__":
    main()
