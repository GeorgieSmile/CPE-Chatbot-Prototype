# chatbot_app.py
from __future__ import annotations

import os
import io
import contextlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local .env if present (no-op on Cloud)
load_dotenv(override=False)

# ---- Import query_data.py (relative import first, then fallback loader) ----
try:
    import query_data as qd
except Exception:
    import importlib.util
    SPEC_PATH = Path(__file__).parent / "query_data.py"
    spec = importlib.util.spec_from_file_location("query_data", SPEC_PATH)
    qd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qd)

# ---- Safe defaults from query_data.py ----
DEFAULT_MODEL = getattr(qd, "DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_K = getattr(qd, "DEFAULT_K", 4)
DEFAULT_MIN_SCORE = getattr(qd, "DEFAULT_MIN_SCORE", 0.55)
DEFAULT_REPLY_LANG = getattr(qd, "DEFAULT_REPLY_LANG", "auto")
DEFAULT_CHROMA_PATH = getattr(qd, "DEFAULT_CHROMA_PATH", "chroma")
DEFAULT_COLLECTION = getattr(qd, "DEFAULT_COLLECTION", "siit-faqs")

# ---- Helpers ----
def read_secret(name: str) -> str | None:
    """Safely read a Streamlit secret without crashing if secrets.toml is missing."""
    try:
        return st.secrets[name]  # raises if no secrets file or key
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
        os.environ["OPENAI_API_KEY"] = key  # make visible to langchain_openai
    return key

def ensure_dir(path_str: str) -> None:
    Path(path_str).mkdir(parents=True, exist_ok=True)

# ---- UI ----
st.set_page_config(page_title="SIIT RAG Assistant", page_icon="🎓", layout="wide")
st.title("🎓 SIIT Academic Support (RAG)")

with st.sidebar:
    st.header("🔑 OpenAI API Key")

    # Show current key source without triggering StreamlitSecretNotFoundError
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

# Main input
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

# ---- Ask flow ----
if ask or demo:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        key = resolve_openai_key()
        if not key:
            st.error("No OpenAI API key found. Paste one in the sidebar or configure Streamlit secrets.")
        else:
            ensure_dir(chroma_path)  # make sure persistence dir exists

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
