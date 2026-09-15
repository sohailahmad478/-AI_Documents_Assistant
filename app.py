````python
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


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="AI Documents Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Documents Assistant")
st.caption(
    "PDF • DOCX • TXT • MD • Google Drive | "
    "RAG + FAISS + Hybrid Search"
)


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_pdf(file_bytes, filename):
    """
    Extract PDF text page-by-page.
    Page numbers are preserved.
    """

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
    """
    Extract text from DOCX.

    DOCX does not reliably provide page numbers,
    therefore page is None.
    """

    document = Document(io.BytesIO(file_bytes))

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(paragraph.text.strip())

    text = "\n".join(paragraphs)

    if not text.strip():
        return []

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }]


def extract_txt(file_bytes, filename):
    """
    Extract TXT text.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "filename": filename,
        "page": None,
    }]


def extract_md(file_bytes, filename):
    """
    Extract Markdown text.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "filename": filename,
        "page": None,
    }]


def extract_document(file_bytes, filename):
    """
    Select extraction function based on extension.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(
            file_bytes,
            filename
        )

    elif extension == ".docx":
        return extract_docx(
            file_bytes,
            filename
        )

    elif extension == ".txt":
        return extract_txt(
            file_bytes,
            filename
        )

    elif extension == ".md":
        return extract_md(
            file_bytes,
            filename
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=800,
    overlap=120
):
    """
    Split text into overlapping word chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_pages):
    """
    Create chunks while preserving:
    filename
    page number
    text
    """

    all_chunks = []

    for item in extracted_pages:

        text_chunks = chunk_text(
            item["text"]
        )

        for chunk in text_chunks:

            all_chunks.append({
                "text": chunk,
                "filename": item["filename"],
                "page": item["page"],
            })

    return all_chunks


# =========================================================
# SENTENCE TRANSFORMERS
# =========================================================

@st.cache_resource
def load_embedding_model():
    """
    Load Sentence Transformer only once.
    """

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# =========================================================
# FAISS VECTOR STORE
# =========================================================

def build_vector_store(chunks):
    """
    Create embeddings and FAISS index.
    """

    if not chunks:
        return None, None

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index, embeddings


# =========================================================
# KEYWORD SEARCH
# =========================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "does",
    "do",
    "did",
    "can",
    "could",
    "please",
    "tell",
    "me",
    "about",
    "from",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "be",
    "as",
    "at",
    "by",
    "into",
}


def important_words(text):
    """
    Extract important words from text.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        word
        for word in words
        if (
            word not in STOP_WORDS
            and len(word) > 2
        )
    ]


def keyword_search(
    question,
    chunks
):
    """
    Score chunks based on
    matching important words.
    """

    query_words = set(
        important_words(question)
    )

    scores = []

    for chunk in chunks:

        chunk_words = set(
            important_words(
                chunk["text"]
            )
        )

        if not query_words:

            score = 0.0

        else:

            matches = (
                query_words
                .intersection(
                    chunk_words
                )
            )

            score = (
                len(matches)
                / len(query_words)
            )

        scores.append(score)

    return np.array(
        scores,
        dtype="float32"
    )


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    question,
    top_k=5
):
    """
    Combine:

    70% semantic search
    30% keyword search
    """

    chunks = (
        st.session_state.chunks
    )

    index = (
        st.session_state.faiss_index
    )

    if not chunks or index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    candidate_k = min(
        max(top_k * 4, 10),
        len(chunks)
    )

    semantic_scores, indices = (
        index.search(
            question_embedding,
            candidate_k
        )
    )

    semantic_scores = (
        semantic_scores[0]
    )

    indices = indices[0]

    keyword_scores = keyword_search(
        question,
        chunks
    )

    results = []

    for semantic_score, idx in zip(
        semantic_scores,
        indices
    ):

        if idx < 0:
            continue

        keyword_score = float(
            keyword_scores[idx]
        )

        hybrid_score = (
            0.70 * float(
                semantic_score
            )
            +
            0.30 * keyword_score
        )

        result = dict(
            chunks[idx]
        )

        result["semantic_score"] = float(
            semantic_score
        )

        result["keyword_score"] = (
            keyword_score
        )

        result["hybrid_score"] = (
            hybrid_score
        )

        results.append(result)

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# GOOGLE DRIVE
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


def get_drive_file_id(url):
    """
    Extract Google Drive file ID.
    """

    patterns = [

        r"/file/d/([a-zA-Z0-9_-]+)",

        r"[?&]id=([a-zA-Z0-9_-]+)",

        r"/open\?id=([a-zA-Z0-9_-]+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:
            return match.group(1)

    return None


def is_drive_folder(url):
    """
    Check whether URL is a Drive folder.
    """

    return "/folders/" in url


def get_filename_from_drive_url(url):
    """
    Try to determine filename from
    a Google Drive URL when possible.
    """

    file_id = get_drive_file_id(url)

    if not file_id:
        return None

    return f"GoogleDrive_{file_id}"


def detect_file_extension(data):
    """
    Detect common document type from file bytes.

    Returns:
        .pdf
        .docx
        .txt
        .md
        None
    """

    # PDF
    if data.startswith(b"%PDF"):
        return ".pdf"

    # DOCX is a ZIP container.
    # Check for DOCX-specific files.
    if data.startswith(b"PK"):

        if (
            b"word/" in data[:20000]
            or b"[Content_Types].xml" in data[:20000]
        ):
            return ".docx"

    # TXT / MD
    try:

        text = data.decode(
            "utf-8",
            errors="ignore"
        )

        if text.strip():

            # Markdown indicators
            markdown_patterns = [
                "# ",
                "## ",
                "### ",
                "- ",
                "* ",
                "```",
                "[",
            ]

            for pattern in markdown_patterns:

                if pattern in text:
                    return ".md"

            return ".txt"

    except Exception:
        pass

    return None


@st.cache_data(show_spinner=False)
def download_drive_file(url):
    """
    Download a public Google Drive file.

    IMPORTANT:
    No fuzzy=True is used because
    some gdown versions do not support it.
    """

    file_id = get_drive_file_id(url)

    if not file_id:

        raise ValueError(
            "Could not find a Google Drive "
            "file ID in this link."
        )

    temp_dir = tempfile.mkdtemp(
        prefix="drive_file_"
    )

    output_path = os.path.join(
        temp_dir,
        "downloaded_file"
    )

    try:

        downloaded_path = gdown.download(
            id=file_id,
            output=output_path,
            quiet=True,
        )

        if (
            not downloaded_path
            or not os.path.exists(
                downloaded_path
            )
        ):

            raise ValueError(
                "Google Drive file could "
                "not be downloaded."
            )

        data = Path(
            downloaded_path
        ).read_bytes()

        extension = detect_file_extension(
            data
        )

        if not extension:

            raise ValueError(
                "Could not determine the "
                "document type. Make sure the "
                "Drive file is PDF, DOCX, TXT "
                "or MD."
            )

        filename = (
            get_filename_from_drive_url(
                url
            )
            + extension
        )

        return data, filename

    finally:

        try:

            for file in Path(
                temp_dir
            ).glob("*"):

                file.unlink(
                    missing_ok=True
                )

            Path(temp_dir).rmdir()

        except Exception:
            pass


def download_drive_folder(url):
    """
    Download supported files from a
    public Google Drive folder.
    """

    folder_dir = tempfile.mkdtemp(
        prefix="drive_folder_"
    )

    try:

        result = gdown.download_folder(
            url,
            output=folder_dir,
            quiet=True,
            use_cookies=False,
        )

        # Some gdown versions return
        # a list, others may return None.
        # We therefore scan the directory.
        files = []

        for path in Path(
            folder_dir
        ).rglob("*"):

            if not path.is_file():
                continue

            extension = (
                path.suffix.lower()
            )

            if extension in SUPPORTED_EXTENSIONS:

                files.append(
                    (
                        path.read_bytes(),
                        path.name
                    )
                )

        return files

    finally:

        # Temporary folder cleanup
        try:

            for path in Path(
                folder_dir
            ).rglob("*"):

                if path.is_file():
                    path.unlink(
                        missing_ok=True
                    )

            for path in sorted(
                Path(folder_dir).rglob("*"),
                reverse=True
            ):

                if path.is_dir():
                    path.rmdir()

            Path(folder_dir).rmdir()

        except Exception:
            pass


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_new_document(
    file_bytes,
    filename,
    source_label="Local upload"
):
    """
    Complete pipeline:

    Document
        ↓
    Extraction
        ↓
    Chunking
        ↓
    Embedding
        ↓
    FAISS
    """

    document_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    # Prevent duplicate processing
    if (
        document_id
        in st.session_state.processed_ids
    ):
        return False

    # -------------------------
    # Extraction
    # -------------------------

    extracted = extract_document(
        file_bytes,
        filename
    )

    if not extracted:

        st.warning(
            f"No text was extracted "
            f"from {filename}."
        )

        return False

    # -------------------------
    # Chunking
    # -------------------------

    new_chunks = create_chunks(
        extracted
    )

    if not new_chunks:

        st.warning(
            f"No chunks were created "
            f"for {filename}."
        )

        return False

    # -------------------------
    # Save document information
    # -------------------------

    st.session_state.documents[
        document_id
    ] = {

        "filename": filename,

        "source": source_label,

        "bytes": len(file_bytes),

        "pages": len(extracted),

        "chunks": len(new_chunks),
    }

    # -------------------------
    # Add chunks
    # -------------------------

    st.session_state.chunks.extend(
        new_chunks
    )

    st.session_state.processed_ids.add(
        document_id
    )

    # -------------------------
    # Build embeddings + FAISS
    # -------------------------

    (
        st.session_state.faiss_index,
        st.session_state.embeddings
    ) = build_vector_store(
        st.session_state.chunks
    )

    return True


# =========================================================
# GROQ
# =========================================================

def get_groq_client():
    """
    Get Groq API key from Streamlit Secrets
    or environment variable.

    RROQ_API_KEY is the primary key name.
    """

    api_key = None

    # Streamlit Cloud Secrets
    try:

        api_key = st.secrets.get(
            "RROQ_API_KEY"
        )

    except Exception:
        pass

    # Environment variable
    if not api_key:

        api_key = os.getenv(
            "RROQ_API_KEY"
        )

    # Optional fallback
    if not api_key:

        try:

            api_key = st.secrets.get(
                "GROQ_API_KEY"
            )

        except Exception:
            pass

    if not api_key:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


def ask_groq(
    question,
    retrieved_chunks
):
    """
    Send the question and retrieved
    document context to Groq.
    """

    client = get_groq_client()

    if client is None:

        raise ValueError(
            "Groq API key not found. "
            "Add RROQ_API_KEY to "
            "Streamlit Secrets."
        )

    # -------------------------
    # Create context
    # -------------------------

    context_parts = []

    for i, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        if item["page"] is not None:

            page_text = (
                f"Page {item['page']}"
            )

        else:

            page_text = (
                "Page not available"
            )

        context_parts.append(

            f"[Source {i}]\n"
            f"Filename: {item['filename']}\n"
            f"{page_text}\n"
            f"Text:\n"
            f"{item['text']}"

        )

    context = "\n\n".join(
        context_parts
    )

    # -------------------------
    # System prompt
    # -------------------------

    system_prompt = """
You are an AI document assistant.

Answer the user's question ONLY
using the provided document context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Do not invent page numbers.

4. Do not invent sources.

5. If the answer is not present
   in the context, say exactly:

"The information is not available
in the provided documents."

6. Give a clear and concise answer.

7. When possible, mention the
   filename or page from the context.
"""

    # -------------------------
    # User prompt
    # -------------------------

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


Answer ONLY using the document
context above.
"""

    # -------------------------
    # Groq request
    # -------------------------

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            },

        ],

        temperature=0.1,

        max_tokens=800,
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Add Documents")

    uploaded_files = st.file_uploader(

        "Upload PDF, DOCX, TXT or MD files",

        type=[
            "pdf",
            "docx",
            "txt",
            "md"
        ],

        accept_multiple_files=True,
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            if st.button(
                f"Process {uploaded_file.name}",
                key=(
                    "process_"
                    + uploaded_file.name
                ),
            ):

                with st.spinner(
                    f"Processing "
                    f"{uploaded_file.name}..."
                ):

                    added = (
                        process_new_document(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                            "Local upload",
                        )
                    )

                if added:

                    st.success(
                        f"{uploaded_file.name} "
                        "processed successfully."
                    )

                else:

                    st.info(
                        f"{uploaded_file.name} "
                        "was already processed."
                    )

    st.divider()

    # =====================================================
    # GOOGLE DRIVE
    # =====================================================

    st.header("☁️ Google Drive")

    drive_url = st.text_input(

        "Paste a public Drive file "
        "or folder link",

        placeholder=(
            "https://drive.google.com/..."
        ),
    )

    if st.button(
        "Load from Google Drive"
    ):

        if not drive_url.strip():

            st.warning(
                "Please paste a "
                "Google Drive link."
            )

        else:

            try:

                with st.spinner(
                    "Loading Google Drive..."
                ):

                    # -------------------------
                    # Folder
                    # -------------------------

                    if is_drive_folder(
                        drive_url
                    ):

                        drive_files = (
                            download_drive_folder(
                                drive_url
                            )
                        )

                        if not drive_files:

                            st.warning(
                                "No supported "
                                "PDF, DOCX, TXT "
                                "or MD files were "
                                "found in the folder."
                            )

                        else:

                            added = 0

                            for (
                                data,
                                filename
                            ) in drive_files:

                                if process_new_document(

                                    data,

                                    filename,

                                    "Google Drive",

                                ):

                                    added += 1

                            st.success(
                                f"Processed "
                                f"{added} new "
                                f"Drive document(s)."
                            )

                    # -------------------------
                    # Single file
                    # -------------------------

                    else:

                        (
                            data,
                            filename
                        ) = download_drive_file(
                            drive_url
                        )

                        extension = (
                            Path(filename)
                            .suffix
                            .lower()
                        )

                        if (
                            extension
                            not in
                            SUPPORTED_EXTENSIONS
                        ):

                            st.error(
                                "This Drive file "
                                "is not supported."
                            )

                        else:

                            added = (
                                process_new_document(
                                    data,
                                    filename,
                                    "Google Drive",
                                )
                            )

                            if added:

                                st.success(
                                    f"{filename} "
                                    "processed."
                                )

                            else:

                                st.info(
                                    f"{filename} "
                                    "was already "
                                    "processed."
                                )

            except Exception as e:

                st.error(
                    "Google Drive loading failed."
                )

                st.exception(e)

    st.divider()

    # =====================================================
    # SEARCH SETTINGS
    # =====================================================

    st.header("⚙️ Search")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    # =====================================================
    # STATISTICS
    # =====================================================

    st.metric(
        "Documents",
        len(
            st.session_state.documents
        )
    )

    st.metric(
        "Created chunks",
        len(
            st.session_state.chunks
        )
    )


# =========================================================
# DOCUMENT INFORMATION
# =========================================================

st.subheader(
    "📊 Extracted Document Information"
)

if st.session_state.documents:

    rows = []

    for document in (
        st.session_state
        .documents
        .values()
    ):

        rows.append({

            "Filename":
                document["filename"],

            "Source":
                document["source"],

            "Size (KB)":
                round(
                    document["bytes"]
                    / 1024,
                    2
                ),

            "Pages / Sections":
                document["pages"],

            "Chunks":
                document["chunks"],

        })

    st.dataframe(
        rows,
        use_container_width=True
    )

    st.info(
        f"Total created chunks: "
        f"**{len(st.session_state.chunks)}**"
    )

    st.caption(
        "Document embeddings are created "
        "when documents are processed and "
        "reused for future questions."
    )

else:

    st.info(
        "Upload a document or load a "
        "public Google Drive file/folder "
        "to begin."
    )


# =========================================================
# ASK QUESTIONS
# =========================================================

st.subheader(
    "💬 Ask Your Documents"
)

question = st.text_input(

    "Enter your question",

    placeholder=(
        "Example: What is the main "
        "objective of this document?"
    ),
)


if st.button(
    "🔎 Search & Answer",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.chunks:

        st.warning(
            "Please add at least "
            "one document first."
        )

    else:

        # -------------------------
        # Hybrid retrieval
        # -------------------------

        with st.spinner(
            "Searching documents..."
        ):

            retrieved = hybrid_search(
                question,
                top_k=top_k
            )

        if not retrieved:

            st.warning(
                "No relevant document "
                "chunks were found."
            )

        else:

            # -------------------------
            # Groq answer
            # -------------------------

            try:

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = ask_groq(
                        question,
                        retrieved
                    )

                st.markdown(
                    "### 🤖 Answer"
                )

                st.write(answer)

            except Exception as e:

                st.error(
                    "Groq request failed."
                )

                st.exception(e)

            # -------------------------
            # Sources
            # -------------------------

            st.markdown(
                "### 📚 Retrieved Sources"
            )

            for i, item in enumerate(
                retrieved,
                start=1
            ):

                if item["page"] is not None:

                    page_text = str(
                        item["page"]
                    )

                else:

                    page_text = (
                        "Not available"
                    )

                with st.expander(

                    f"{i}. "
                    f"{item['filename']} "
                    f"| Page: "
                    f"{page_text}"

                ):

                    st.write(

                        f"**Semantic score:** "
                        f"{item['semantic_score']:.3f}"
                        "\n\n"

                        f"**Keyword score:** "
                        f"{item['keyword_score']:.3f}"
                        "\n\n"

                        f"**Hybrid score:** "
                        f"{item['hybrid_score']:.3f}"

                    )

                    st.write(
                        item["text"]
                    )


# =========================================================
# CLEAR DOCUMENTS
# =========================================================

st.divider()

if st.button(
    "🗑️ Clear All Documents"
):

    st.session_state.documents = {}

    st.session_state.chunks = []

    st.session_state.embeddings = None

    st.session_state.faiss_index = None

    st.session_state.processed_ids = set()

    st.rerun()
````
