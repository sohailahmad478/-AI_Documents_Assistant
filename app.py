import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Documents Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Documents Assistant")
st.caption("PDF • DOCX • TXT • MD • Google Drive | RAG + FAISS + Hybrid Search")


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()

if "model" not in st.session_state:
    st.session_state.model = None


# -----------------------------
# Document extraction functions
# -----------------------------
def extract_pdf(file_bytes, filename):
    """Extract PDF text page-by-page and preserve page numbers."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text.strip(),
                "filename": filename,
                "page": page_number,
            })

    return pages


def extract_docx(file_bytes, filename):
    """Extract DOCX paragraphs. DOCX does not reliably expose page numbers."""
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "filename": filename,
        "page": None,
    }]


def extract_txt(file_bytes, filename):
    """Extract TXT text."""
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "filename": filename,
        "page": None,
    }]


def extract_md(file_bytes, filename):
    """Extract Markdown text as plain text."""
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "filename": filename,
        "page": None,
    }]


def extract_document(file_bytes, filename):
    """Select the correct extractor based on file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    elif extension == ".docx":
        return extract_docx(file_bytes, filename)
    elif extension == ".txt":
        return extract_txt(file_bytes, filename)
    elif extension == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {extension}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=800, overlap=120):
    """Split text into overlapping word chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_pages):
    """Create chunks while keeping filename and page metadata."""
    all_chunks = []

    for item in extracted_pages:
        text_chunks = chunk_text(item["text"])

        for chunk in text_chunks:
            all_chunks.append({
                "text": chunk,
                "filename": item["filename"],
                "page": item["page"],
            })

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the embedding model only once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_store(chunks):
    """Create document embeddings once and build the FAISS index."""
    if not chunks:
        return None, None

    model = load_embedding_model()

    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "on", "for", "and", "or", "with", "what", "which", "who", "how",
    "why", "when", "where", "does", "do", "did", "can", "could",
    "please", "tell", "me", "about", "from", "this", "that", "these",
    "those", "it", "its", "be", "as", "at", "by", "into",
}


def important_words(text):
    """Get simple important words for keyword matching."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_search(question, chunks):
    """Score each chunk using simple keyword overlap."""
    query_words = set(important_words(question))
    scores = []

    for chunk in chunks:
        chunk_words = set(important_words(chunk["text"]))
        if not query_words:
            score = 0.0
        else:
            score = len(query_words.intersection(chunk_words)) / len(query_words)

        scores.append(score)

    return np.array(scores, dtype="float32")


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=5):
    """Combine FAISS semantic similarity and keyword matching."""
    chunks = st.session_state.chunks
    index = st.session_state.faiss_index

    if not chunks or index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Search more candidates first, then combine with keyword scores.
    candidate_k = min(max(top_k * 4, 10), len(chunks))
    semantic_scores, indices = index.search(question_embedding, candidate_k)

    semantic_scores = semantic_scores[0]
    indices = indices[0]

    keyword_scores_all = keyword_search(question, chunks)

    candidates = []

    for semantic_score, idx in zip(semantic_scores, indices):
        if idx < 0:
            continue

        keyword_score = float(keyword_scores_all[idx])

        # 70% semantic + 30% keyword
        hybrid_score = (0.70 * float(semantic_score)) + (0.30 * keyword_score)

        result = dict(chunks[idx])
        result["semantic_score"] = float(semantic_score)
        result["keyword_score"] = keyword_score
        result["hybrid_score"] = hybrid_score

        candidates.append(result)

    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Google Drive helpers
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_drive_file_id(url):
    """Get a Google Drive file ID from common Drive URLs."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def is_drive_folder(url):
    return "/folders/" in url


@st.cache_data(show_spinner=False)
def download_drive_file(url):
    """Download one public Google Drive file."""
    file_id = get_drive_file_id(url)

    if not file_id:
        raise ValueError("Could not find a Google Drive file ID in the link.")

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name

    try:
        downloaded = gdown.download(
            id=file_id,
            output=temp_path,
            quiet=True,
            fuzzy=True,
        )

        if not downloaded or not os.path.exists(downloaded):
            raise ValueError("Google Drive file could not be downloaded.")

        data = Path(downloaded).read_bytes()
        return data, Path(downloaded).name

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_drive_folder(url):
    """Download files from a public Google Drive folder using gdown."""
    folder_dir = tempfile.mkdtemp(prefix="drive_docs_")

    gdown.download_folder(
        url,
        output=folder_dir,
        quiet=True,
        use_cookies=False,
    )

    files = []

    for path in Path(folder_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append((path.read_bytes(), path.name))

    return files


# -----------------------------
# Processing pipeline
# -----------------------------
def process_new_document(file_bytes, filename, source_label=None):
    """Extract, chunk and add one document to the reusable vector store."""
    document_id = hashlib.sha256(file_bytes).hexdigest()

    if document_id in st.session_state.processed_ids:
        return False

    extracted = extract_document(file_bytes, filename)

    if not extracted:
        st.warning(f"No text was extracted from {filename}.")
        return False

    new_chunks = create_chunks(extracted)

    if not new_chunks:
        st.warning(f"No chunks were created for {filename}.")
        return False

    # Add document information.
    st.session_state.documents[document_id] = {
        "filename": filename,
        "source": source_label or "Local upload",
        "bytes": len(file_bytes),
        "pages": len(extracted),
        "chunks": len(new_chunks),
    }

    # Add chunks.
    st.session_state.chunks.extend(new_chunks)
    st.session_state.processed_ids.add(document_id)

    # Rebuild the FAISS store only when a NEW document is added.
    st.session_state.faiss_index, st.session_state.embeddings = build_vector_store(
        st.session_state.chunks
    )

    return True


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    # User requested RROQ_API_KEY. GROQ_API_KEY is also supported as a convenience.
    api_key = st.secrets.get("RROQ_API_KEY", None)

    if not api_key:
        api_key = os.getenv("RROQ_API_KEY")

    if not api_key:
        api_key = st.secrets.get("GROQ_API_KEY", None)

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def ask_groq(question, retrieved_chunks):
    """Ask Groq to answer only from retrieved document context."""
    client = get_groq_client()

    if client is None:
        raise ValueError(
            "Groq API key not found. Add RROQ_API_KEY to Streamlit Secrets."
        )

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {item['page']}"
            if item["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {i}]\n"
            f"Filename: {item['filename']}\n"
            f"{page_text}\n"
            f"Text:\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are an AI document assistant.

Answer the user's question ONLY using the provided document context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "The information is not available in the provided documents."
3. Do not invent facts, sources, page numbers, or details.
4. Give a clear and concise answer.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer only from the document context above."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    return response.choices[0].message.content


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📄 Add Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if st.button(
                f"Process {uploaded_file.name}",
                key=f"process_{uploaded_file.name}",
            ):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    process_new_document(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        "Local upload",
                    )
                st.success(f"{uploaded_file.name} processed.")

    st.divider()

    st.header("☁️ Google Drive")
    drive_url = st.text_input(
        "Paste a public Drive file or folder link",
        placeholder="https://drive.google.com/...",
    )

    if st.button("Load from Google Drive"):
        if not drive_url.strip():
            st.warning("Please paste a Google Drive link.")
        else:
            try:
                with st.spinner("Loading Google Drive document(s)..."):
                    if is_drive_folder(drive_url):
                        drive_files = download_drive_folder(drive_url)

                        if not drive_files:
                            st.warning(
                                "No supported PDF, DOCX, TXT or MD files were found."
                            )
                        else:
                            added = 0
                            for data, filename in drive_files:
                                if process_new_document(
                                    data,
                                    filename,
                                    "Google Drive",
                                ):
                                    added += 1

                            st.success(f"Processed {added} new Drive document(s).")
                    else:
                        data, filename = download_drive_file(drive_url)

                        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                            st.error(
                                "This Drive file is not PDF, DOCX, TXT or MD."
                            )
                        else:
                            added = process_new_document(
                                data,
                                filename,
                                "Google Drive",
                            )
                            if added:
                                st.success(f"{filename} processed.")
                            else:
                                st.info(f"{filename} was already processed.")

            except Exception as e:
                st.error(
                    "Google Drive loading failed. Make sure the file/folder is "
                    f"publicly accessible.\n\nDetails: {e}"
                )

    st.divider()

    st.header("⚙️ Search")
    top_k = st.slider("Retrieved chunks", 1, 10, 5)

    st.divider()

    st.metric("Documents", len(st.session_state.documents))
    st.metric("Created chunks", len(st.session_state.chunks))


# -----------------------------
# Main document information
# -----------------------------
st.subheader("📊 Extracted Document Information")

if st.session_state.documents:
    rows = []

    for item in st.session_state.documents.values():
        rows.append({
            "Filename": item["filename"],
            "Source": item["source"],
            "Size (KB)": round(item["bytes"] / 1024, 2),
            "Pages / Sections": item["pages"],
            "Chunks": item["chunks"],
        })

    st.dataframe(rows, use_container_width=True)

    st.info(
        f"Total created chunks: **{len(st.session_state.chunks)}**. "
        "Document embeddings are created once and reused for questions."
    )
else:
    st.info(
        "Upload a document or load a public Google Drive file/folder to begin."
    )


# -----------------------------
# Question answering
# -----------------------------
st.subheader("💬 Ask Your Documents")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the main objective of this document?",
)

if st.button("🔎 Search & Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.chunks:
        st.warning("Please add at least one document first.")
    else:
        with st.spinner("Searching documents..."):
            retrieved = hybrid_search(question, top_k=top_k)

        if not retrieved:
            st.warning("No relevant document chunks were found.")
        else:
            try:
                with st.spinner("Generating answer with Groq..."):
                    answer = ask_groq(question, retrieved)

                st.markdown("### 🤖 Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Groq request failed: {e}")

            st.markdown("### 📚 Retrieved Sources")

            for i, item in enumerate(retrieved, start=1):
                page_text = (
                    str(item["page"])
                    if item["page"] is not None
                    else "Not available"
                )

                with st.expander(
                    f"{i}. {item['filename']} | Page: {page_text}"
                ):
                    st.write(
                        f"**Semantic score:** {item['semantic_score']:.3f}  \n"
                        f"**Keyword score:** {item['keyword_score']:.3f}  \n"
                        f"**Hybrid score:** {item['hybrid_score']:.3f}"
                    )
                    st.write(item["text"])


# -----------------------------
# Clear data
# -----------------------------
st.divider()

if st.button("🗑️ Clear All Documents"):
    st.session_state.documents = {}
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.faiss_index = None
    st.session_state.processed_ids = set()
    st.rerun()
