import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import build_vector_store, answer_question, get_embedding_model_name

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

st.set_page_config(
    page_title="EduLens | Study smarter",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Styling --------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(70, 105, 220, .13), transparent 34%),
          radial-gradient(ellipse at 90% 12%, rgba(22, 163, 154, .10), transparent 28%);
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    .hero {
        padding: 1.55rem 1.7rem;
        border: 1px solid rgba(120, 140, 190, .25);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(55, 83, 170, .16), rgba(22, 163, 154, .08));
        margin-bottom: 1.25rem;
    }
    .eyebrow {
        color: #6c91f4; font-size: .76rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: .13em;
    }
    .hero-title { font-family: 'Manrope', sans-serif; font-size: 2.25rem; font-weight: 800; margin: .25rem 0; }
    .hero-sub { color: var(--text-color); opacity: .72; font-size: 1rem; max-width: 700px; }
    .panel {
        border: 1px solid rgba(120, 140, 190, .23);
        border-radius: 17px; padding: 1rem 1.1rem;
        background: rgba(120, 140, 190, .045);
    }
    .small-muted { color: var(--text-color); opacity: .65; font-size: .83rem; }
    .step-card {
        border: 1px solid rgba(120, 140, 190, .22);
        border-radius: 15px; padding: .9rem; min-height: 115px;
        background: rgba(120, 140, 190, .045);
    }
    .step-num { color: #6c91f4; font-weight: 800; font-size: .75rem; letter-spacing: .1em; }
    .step-title { font-weight: 700; margin: .25rem 0; }
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(120, 140, 190, .16);
        border-radius: 16px; padding: .4rem .65rem;
        margin-bottom: .65rem;
    }
    div.stButton > button {
        border-radius: 11px; font-weight: 600;
        border: 1px solid rgba(120, 140, 190, .35);
    }
    div.stButton > button[kind="primary"] {
        border: 0; background: linear-gradient(100deg, #4268d5, #278e9b);
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 140, 190, .18);
    }
    .status-dot {
        display:inline-block; width:8px; height:8px; border-radius:50%;
        background:#20b486; margin-right:7px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Session state --------------------------------
defaults = {
    "vector_store": None,
    "document_names": [],
    "document_count": 0,
    "chunk_count": 0,
    "chat_history": [],
    "pending_question": "",
    "indexed_fingerprint": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------ Sidebar --------------------------------------
with st.sidebar:
    st.markdown("## 🎓 EduLens")
    st.caption("Your academic document assistant")
    st.divider()

    st.markdown("### 01 · Knowledge base")
    uploaded_files = st.file_uploader(
        "Add your study PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, syllabi, or previous-year question papers.",
    )

    with st.expander("Retrieval settings", expanded=False):
        chunk_size = st.slider("Chunk size", 400, 1600, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 300, 150, 25)
        top_k = st.slider("Passages to retrieve", 1, 8, 4)

    if st.button("⚡ Build knowledge base", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            file_fingerprint = hashlib.sha256(
                b"".join(
                    f.name.encode("utf-8") + f.getvalue()
                    for f in uploaded_files
                )
            ).hexdigest()
            try:
                with st.spinner("Reading PDFs, creating embeddings, and building the FAISS index..."):
                    store, chunk_count = build_vector_store(
                        uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                st.session_state.vector_store = store
                st.session_state.document_names = [f.name for f in uploaded_files]
                st.session_state.document_count = len(uploaded_files)
                st.session_state.chunk_count = chunk_count
                st.session_state.chat_history = []
                st.session_state.indexed_fingerprint = file_fingerprint
                st.success(f"Knowledge base ready · {chunk_count} chunks indexed")
            except Exception as exc:
                st.error(f"Could not process documents: {exc}")

    if st.session_state.vector_store is not None:
        st.markdown(
            '<span class="status-dot"></span><b>Knowledge base ready</b>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{st.session_state.document_count} document(s) · "
            f"{st.session_state.chunk_count} chunks"
        )
        for filename in st.session_state.document_names:
            st.markdown(f"📄 `{filename}`")

    st.divider()
    st.markdown("### 02 · Gemini connection")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        st.success("Gemini API key loaded from .env")
    else:
        st.error("Gemini API key not found. Add GOOGLE_API_KEY to your .env file.")
    st.caption("The API key is read only from .env and is never requested on this page.")
    st.caption("Embeddings run locally with Sentence Transformers.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Reset knowledge base", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.document_names = []
        st.session_state.document_count = 0
        st.session_state.chunk_count = 0
        st.session_state.chat_history = []
        st.session_state.indexed_fingerprint = None
        st.rerun()

    st.divider()
    st.markdown("**Model configuration**")
    st.caption(f"Embedding model: `{get_embedding_model_name()}`")
    st.caption(f"Answer model: `{os.getenv('GEMINI_CHAT_MODEL', 'gemini-3.8-flash')}`")

# ------------------------------ Main content ---------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered learning workspace</div>
      <div class="hero-title">Understand your study material.</div>
      <div class="hero-sub">
        Upload academic PDFs, ask questions in plain language, and get answers
        grounded in your documents with page-level source references.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vector_store is None:
    st.markdown("### Your study workspace")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="step-card"><div class="step-num">STEP 01</div>'
            '<div class="step-title">Upload documents</div>'
            '<div class="small-muted">Add PDFs such as notes, textbooks, or question papers.</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="step-card"><div class="step-num">STEP 02</div>'
            '<div class="step-title">Build your library</div>'
            '<div class="small-muted">EduLens extracts text and indexes its meaning locally.</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="step-card"><div class="step-num">STEP 03</div>'
            '<div class="step-title">Ask and learn</div>'
            '<div class="small-muted">Get contextual explanations with document references.</div></div>',
            unsafe_allow_html=True,
        )

    st.info("Start by uploading PDFs in the left sidebar, then select **Build knowledge base**.")
    st.markdown("#### Example questions")
    examples = [
        "Explain the main concepts in this material.",
        "What are the important definitions in this chapter?",
        "Compare the concepts discussed in the document.",
    ]
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, use_container_width=True, key=f"example_{i}"):
                st.session_state.pending_question = example
                st.rerun()
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", st.session_state.document_count)
    m2.metric("Indexed chunks", st.session_state.chunk_count)
    m3.metric("Retrieved per question", top_k)

    st.markdown("### Ask your documents")
    st.caption("Answers are generated from retrieved passages. If the documents do not contain the answer, EduLens will say so.")

# Render conversation history
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander(f"View {len(item['sources'])} retrieved source passage(s)"):
                for source in item["sources"]:
                    st.markdown(f"**{source['source']} · Page {source['page']}**")
                    st.write(source["excerpt"])
                    st.divider()

# Chat input and example prompt handoff
question = st.chat_input(
    "Ask something from your uploaded study material...",
    disabled=st.session_state.vector_store is None,
)
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""

if question:
    if st.session_state.vector_store is None:
        st.warning("Build a knowledge base before asking questions.")
    elif not api_key:
        st.warning("Add GOOGLE_API_KEY to the .env file in the project folder, then restart Streamlit.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your study material..."):
                try:
                    result = answer_question(
                        vector_store=st.session_state.vector_store,
                        question=question,
                        top_k=top_k,
                        api_key=api_key,
                    )
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander(f"View {len(result['sources'])} retrieved source passage(s)", expanded=True):
                            for source in result["sources"]:
                                st.markdown(f"**{source['source']} · Page {source['page']}**")
                                st.write(source["excerpt"])
                                st.divider()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")
                    st.caption("Check your Gemini API key, model availability, and internet connection.")
