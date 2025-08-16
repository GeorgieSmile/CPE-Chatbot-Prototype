# chatbot_app.py
from __future__ import annotations

# --- Patch sqlite for Chroma (Streamlit Cloud often has old sqlite3) ---
# Requires: pysqlite3-binary in requirements.txt
try:
    import sqlite3
    ver = tuple(map(int, sqlite3.sqlite_version.split(".")))
    if ver < (3, 35, 0):
        import pysqlite3  # type: ignore
        import sys
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass  # If patch fails, Chroma may still try its own fallback

import os
import io
import contextlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local .env if present (safe no-op on Cloud)
load_dotenv(override=False)

# --- Import query_data (relative, with safe fallback loader) ---
try:
    import query_data as qd
except Exception:
    import importlib.util
    SPEC_PATH = Path(__file__).parent / "query_data.py"
    spec = importlib.util.spec_from_file_location("query_data", SPEC_PATH)
    qd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qd)

# --- Defaults from query_data (with safe fallbacks) ---
DEFAULT_MODEL = getattr(qd, "DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_K = getattr(qd, "DEFAULT_K", 4)
DEFAULT_MIN_SCORE = getattr(qd, "DEFAULT_MIN_SCORE", 0.55)
DEFAULT_REPLY_LANG = getattr(qd, "DEFAULT_REPLY_LANG", "auto")
DEFAULT_CHROMA_PATH = getattr(qd, "DEFAULT_CHROMA_PATH", "chroma")
DEFAULT_COLLECTION = getattr(qd, "DEFAULT_COLLECTION", "siit-faqs")

# --- Helpers ---
def read_secret(name: str) -> str | None:
    """Safely read a Streamlit secret without raising when secrets.toml is missing."""
    try:
        return st.secrets[name]  # type: ignore[attr-defined]
    except Exception:
        return None

def resolve_openai_key() -> str | None:
    """
    Priority:
      1) Streamlit secrets
      2) Environment variable
      3) Session (pasted by user)
    """
    key = read_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key  # surface to langchain_openai
    return key

def ensure_dir(path_str: str) -> None:
    Path(path_str).mkdir(parents=True, exist_ok=True)

# --- UI ---
st.set_page_config(page_title="SIIT RAG Assistant", page_icon="🎓", layout="wide")
st.title("🎓 SIIT Academic Support (RAG)")

with st.sidebar:
    st.header("🔑 OpenAI API Key")

    # Non-crashing key source indicator
    if read_secret("OPENAI_API_KEY"):
        key_source = "Streamlit Secrets"
    elif os.getenv("OPENAI_API_KEY"):
        key_source = "Environment"
    elif st.session_state.get("OPENAI_API_KEY"):
        key_source = "Session"
    else:
        key_source = "None"
    st.caption(f"Key source: **{key_source}**")

    pasted = st.text_input("Enter your OpenAI API key", type="password", placeholder="sk-...")
    if pasted:
        st.session_state["OPENAI_API_KEY"] = pasted
        os.environ["OPENAI_API_KEY"] = pasted
        st.success("API key saved for this session.")

    st.divider()
    st.header("⚙️ Settings")
    model_name = st.text_input("OpenAI Chat model", value=DEFAULT_MODEL)
    k = st.slider("Top-K documents", 1, 10, DEFAULT_K)
    min_score = st.slider("Min relevance (vector-only)", 0.0, 1.0, DEFAULT_MIN_SCORE, 0.01)
    use_bm25 = st.checkbox("Use BM25 (Ensemble)", value=True)
    reply_lang = st.selectbox("Answer language", ["auto", "th", "en"],
                              index=["auto", "th", "en"].index(DEFAULT_REPLY_LANG))

    st.divider()
    chroma_path = st.text_input("Chroma path", value=DEFAULT_CHROMA_PATH)
    collection = st.text_input("Collection name", value=DEFAULT_COLLECTION)

    st.divider()
    show_debug = st.checkbox("Show retrieval debug panel", value=False)

question = st.text_area(
    "Ask your question",
    height=120,
    placeholder="เช่น ทำบัตรนักศึกษาหายต้องทำอย่างไร? / How do I request a transcript?",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    ask = st.button("Ask")
with col2:
    demo = st.button("Try sample (Transcript)")
with col3:
    clear = st.button("Clear session key")

if clear:
    st.session_state.pop("OPENAI_API_KEY", None)
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    st.toast("Session API key cleared.", icon="🧹")

if demo:
    question = "How do I request a SIIT Transcript/Certificate document?"
    st.info("Using sample question.")

# --- Handle ask/demo ---
if ask or demo:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        key = resolve_openai_key()
        if not key:
            st.error("No OpenAI API key found. Paste one in the sidebar or configure Streamlit secrets.")
        else:
            ensure_dir(chroma_path)

            # Optional retrieval debug panel (does not affect final answer)
            if show_debug:
                try:
                    db_dbg = qd.get_db(chroma_path, collection)
                    retriever_dbg = qd.build_retriever(db_dbg, use_bm25=use_bm25, k=k)
                    expanded_q = qd.expand_query(question.strip())

                    docs_dbg = []
                    scores_dbg = None

                    # If it's the ensemble path, there are no numeric scores
                    try:
                        from langchain.retrievers import EnsembleRetriever  # type: ignore
                        is_ensemble = isinstance(retriever_dbg, EnsembleRetriever)
                    except Exception:
                        is_ensemble = False

                    if is_ensemble:
                        docs_dbg = retriever_dbg.invoke(expanded_q)
                    else:
                        results_dbg = qd.similarity_search_with_scores(db_dbg, expanded_q, k=k)
                        docs_dbg = [doc for doc, _s in results_dbg]
                        scores_dbg = [float(_s) for _d, _s in results_dbg]

                    with st.expander("🔎 Debug: retrieved documents & query", expanded=False):
                        st.write("**Expanded query used for retrieval:**", expanded_q)
                        if not docs_dbg:
                            st.info("No documents were retrieved.")
                        else:
                            for i, doc in enumerate(docs_dbg, start=1):
                                md = (doc.metadata or {})
                                title = md.get("title", "")
                                section = md.get("section", "")
                                source = md.get("source", "")
                                score = f"{scores_dbg[i-1]:.3f}" if scores_dbg and i-1 < len(scores_dbg) else "—"
                                st.markdown(f"**{i}. {title} › {section}** ({source}) — score: {score}")
                                excerpt = (doc.page_content or "").strip().replace("\n", " ")
                                if len(excerpt) > 600:
                                    excerpt = excerpt[:600] + " ..."
                                st.code(excerpt, language="markdown")
                except Exception as e:
                    st.caption(f"Debug panel error (safe to ignore): {e}")

            # Capture printed Markdown from qd.run_query(...)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    qd.run_query(
                        query_text=question.strip(),
                        chroma_path=chroma_path,
                        collection=collection,
                        model_name=model_name,
                        k=k,
                        min_score=min_score,
                        use_bm25=use_bm25,
                        reply_lang=reply_lang,
                    )
                output_md = buf.getvalue().strip()
                if not output_md:
                    st.warning("No response produced.")
                elif output_md == "Ask CPE/DE Secretary for more information":
                    st.info(output_md)
                else:
                    st.markdown(output_md, unsafe_allow_html=False)
            except Exception as e:
                st.exception(e)
