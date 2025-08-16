#!/usr/bin/env python3
"""
Query a Chroma vector DB built from Markdown FAQs.

Adds language-mirroring:
- Auto-detect Thai and answer in Thai when the question is Thai.
- CLI: --reply-lang auto|th|en (default: auto)

Other features kept:
- Ensemble (Vector + BM25) retrieval with retriever.invoke()
- Minimal query expansion for campus terms
- Temperature=0, clean Markdown with Sources
"""

import argparse
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

# ----------------------------
# Defaults (override via CLI)
# ----------------------------
DEFAULT_CHROMA_PATH = "chroma"
DEFAULT_COLLECTION = "siit-faqs"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_K = 4                 # slightly higher recall
DEFAULT_MIN_SCORE = 0.55      # used only in pure vector mode
DEFAULT_REPLY_LANG = "auto"   # auto | th | en

PROMPT_TEMPLATE = """
You are an academic support assistant for SIIT students.
Use ONLY the provided context to answer.
{lang_instruction}

Context:
{context}

---
Student question: {question}

Rules:
- If the answer is in context: reply concisely in bullet points, include any relevant links and a short "Next steps".
- If NOT in context: reply exactly with:
  Ask CPE/DE Secretary for more information
- Always end with a short "Sources" section listing the FAQ titles/sections.

Return format (Markdown):
- **Answer:** ...
- **Next steps:** ...
- **Sources:** • <title> › <section> (file)
""".strip()


def require_openai_key():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... "
            "or export it in your shell."
        )


def get_db(chroma_path: str, collection: str) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name=collection,
    )
    return db


def get_all_documents_from_db(db: Chroma) -> List[Document]:
    raw = db.get(include=["metadatas", "documents"])
    docs = []
    for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def build_retriever(db: Chroma, use_bm25: bool, k: int):
    vector_retriever = db.as_retriever(search_kwargs={"k": k})

    if not use_bm25:
        return vector_retriever

    docs = get_all_documents_from_db(db)
    if not docs:
        return vector_retriever

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k

    # Favor vectors a bit for paraphrases
    return EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.6, 0.4])


def expand_query(query: str) -> str:
    """
    Minimal bilingual + wording expansion. We APPEND hints; we don't replace.
    """
    expansions = []

    # English wording harmonization
    if "transcript" in query.lower() and "certificate" not in query.lower():
        expansions.append("Transcript/Certificate")
    if "power of attorney" in query.lower():
        expansions.append("Power of Attorney form")
    if "student card" in query.lower():
        expansions.append("student card / citizen ID / passport")

    # Thai -> English hints
    th_map = {
        "ใบมอบอำนาจ": "Power of Attorney",
        "บัตรนิสิต": "student card",
        "นักศึกษา": "student",
        "ทำบัตร": "student card",
        "ลงทะเบียน": "registration",
        "โปรแกรม": "program (major)",
        "ผลคะแนนอังกฤษ": "English score",
        "ทะเบียน": "registrar",
    }
    for th, en in th_map.items():
        if th in query and en.lower() not in query.lower():
            expansions.append(en)

    if expansions:
        return f"{query} ({', '.join(expansions)})"
    return query


def format_sources(docs: List[Document]) -> str:
    seen = set()
    lines = []
    for d in docs:
        md = d.metadata or {}
        t = md.get("title", "") or ""
        s = md.get("section", "") or ""
        src = md.get("source", "") or ""
        line = f"• {t} › {s} ({src})".strip()
        if line not in seen:
            lines.append(line)
            seen.add(line)
    return "\n".join(lines) if lines else "• (no sources metadata)"


def similarity_search_with_scores(db: Chroma, query: str, k: int) -> List[Tuple[Document, float]]:
    return db.similarity_search_with_relevance_scores(query, k=k)


# ----------------------------
# Language helpers
# ----------------------------
THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")

def is_thai(text: str) -> bool:
    return bool(THAI_CHAR_RE.search(text or ""))

def make_lang_instruction(reply_lang: str, question: str) -> str:
    """
    Build a short instruction for the LLM to respond in Thai or English.
    """
    if reply_lang == "th":
        return "Reply in Thai."
    if reply_lang == "en":
        return "Reply in English."
    # auto
    return "Reply in Thai." if is_thai(question) else "Reply in English."


def run_query(
    query_text: str,
    chroma_path: str,
    collection: str,
    model_name: str,
    k: int,
    min_score: float,
    use_bm25: bool,
    reply_lang: str,
):
    # 1) Build DB and retriever
    db = get_db(chroma_path, collection)
    retriever = build_retriever(db, use_bm25=use_bm25, k=k)

    # 2) Lightweight expansion (helps BM25 + vectors)
    q = expand_query(query_text)

    # 3) Retrieve
    if isinstance(retriever, EnsembleRetriever):
        docs: List[Document] = retriever.invoke(q)
        if not docs:
            print("Ask CPE/DE Secretary for more information")
            return
    else:
        results = similarity_search_with_scores(db, q, k=k)
        if not results or results[0][1] < min_score:
            if q != query_text:
                results = similarity_search_with_scores(db, q, k=k)
            if not results or results[0][1] < min_score:
                print("Ask CPE/DE Secretary for more information")
                return
        docs = [doc for doc, _ in results]

    # 4) Compose prompt context + language instruction
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
    lang_instruction = make_lang_instruction(reply_lang, query_text)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
        lang_instruction=lang_instruction,
    )

    # 5) Model call
    model = ChatOpenAI(model=model_name, temperature=0, max_tokens=500)
    resp = model.invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    # 6) Ensure sources rendered
    src_md = format_sources(docs)
    if "**Sources:**" not in answer:
        answer = f"{answer.rstrip()}\n\n**Sources:**\n{src_md}"

    print(answer)


def parse_args():
    p = argparse.ArgumentParser(description="Query SIIT FAQ RAG index.")
    p.add_argument("question", nargs="+", help="Student question")
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="Chroma persistence folder")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model (e.g., gpt-4o-mini)")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K documents to retrieve")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Min relevance (pure vector only)")
    p.add_argument("--no-bm25", action="store_true", help="Disable BM25 and use pure vector search")
    p.add_argument("--reply-lang", choices=["auto", "th", "en"], default=DEFAULT_REPLY_LANG,
                   help="Answer language: auto (mirror), th, or en")
    return p.parse_args()


def main():
    require_openai_key()
    args = parse_args()
    question = " ".join(args.question).strip()
    if not question:
        print("Please provide a question.")
        return

    run_query(
        query_text=question,
        chroma_path=args.chroma_path,
        collection=args.collection,
        model_name=args.model,
        k=args.k,
        min_score=args.min_score,
        use_bm25=not args.no_bm25,
        reply_lang=args.reply_lang,
    )


if __name__ == "__main__":
    main()
