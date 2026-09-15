# 📚 AI Documents Assistant

A simple **Streamlit RAG document assistant** that can read:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Public Google Drive PDF, DOCX, TXT and MD files

The application uses:

**Document → Extraction → Chunking → Sentence Transformers → FAISS + Keyword Search → Hybrid Retrieval → Groq**

## 1. Features

### Local documents
Upload one or more:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

The app extracts the text and shows:

- filename
- source
- file size
- pages/sections
- number of chunks

PDF page numbers are preserved. DOCX, TXT and MD files do not have reliable page numbers, so their page value is shown as unavailable.

### Text chunking

Extracted text is split into overlapping chunks.

Default:

- chunk size: 800 words
- overlap: 120 words

Every chunk keeps:

- filename
- page number when available
- text

### Sentence Transformers

The application uses:

`all-MiniLM-L6-v2`

Document chunks are embedded **when a new document is processed**, not every time the user asks a question.

The embeddings are kept in Streamlit session state.

### FAISS

FAISS stores the document vectors and performs semantic similarity search.

For a question:

1. The question is converted into an embedding.
2. FAISS retrieves candidate chunks.
3. Keyword matching scores the same chunks.
4. Semantic and keyword scores are combined.
5. The best chunks are sent to Groq.

The document embeddings are not recreated for every question.

### Hybrid search

The current simple scoring is:

```text
Hybrid Score = 70% Semantic Score + 30% Keyword Score
```

This makes the application useful for both:

- meaning-based questions
- exact/important word matching

### Groq

Groq receives:

- the user's question
- the retrieved document chunks

The system prompt tells Groq to answer **only from the supplied context**.

If the information is not available, it must say:

> The information is not available in the provided documents.

The current model is:

```text
openai/gpt-oss-20b
```

## 2. Project files

Only three files are required:

```text
document-assistant/
│
├── app.py
├── requirements.txt
└── README.md
```

## 3. Install locally

Create a virtual environment if desired, then run:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

## 4. Groq API key

Do **NOT** put the API key inside `app.py`.

For Streamlit Cloud, open:

**App → Settings → Secrets**

Add:

```toml
RROQ_API_KEY = "your_groq_api_key_here"
```

The application also accepts `GROQ_API_KEY`, but `RROQ_API_KEY` is the primary key name used by this project.

For local development, you can use an environment variable:

```text
RROQ_API_KEY=your_groq_api_key_here
```

## 5. Google Drive

The Google Drive feature is designed for **publicly accessible** files/folders.

Paste a Google Drive file or folder link in the sidebar.

Supported files:

```text
PDF
DOCX
TXT
MD
```

### Important

For a Drive file/folder to work without OAuth, it should be accessible through its public/shared link.

Private Drive files that require your personal Google login may not be downloadable by this simple version.

## 6. How the RAG pipeline works

```text
                 LOCAL UPLOAD
                      │
                      ▼
              PDF / DOCX / TXT / MD
                      │
                      ▼
              TEXT EXTRACTION
                      │
                      ▼
             OVERLAPPING CHUNKS
                      │
                      ▼
          SENTENCE TRANSFORMERS
                      │
                      ▼
                 EMBEDDINGS
                      │
                      ▼
                    FAISS
                      │
                      │
USER QUESTION ────────┤
      │               │
      ▼               ▼
Question Embedding   Keywords
      │               │
      └───────┬───────┘
              ▼
         HYBRID SEARCH
              │
              ▼
       TOP RELEVANT CHUNKS
              │
              ▼
             GROQ
              │
              ▼
          FINAL ANSWER
              │
              ▼
       RETRIEVED SOURCES
```

Google Drive follows the same pipeline:

```text
Google Drive
     ↓
Download supported files
     ↓
Same extraction function
     ↓
Same chunking
     ↓
Same embeddings
     ↓
Same FAISS index
     ↓
Same hybrid search
```

## 7. Important optimization

The application keeps these objects in Streamlit session state:

```python
st.session_state.chunks
st.session_state.embeddings
st.session_state.faiss_index
st.session_state.processed_ids
```

This means:

- adding a new document triggers processing
- an already processed document is skipped
- document embeddings are reused for future questions
- asking another question does not re-embed all documents

Only the **new question** needs to be embedded for semantic search.

The embedding model is also loaded with:

```python
@st.cache_resource
```

so it is reused instead of loaded repeatedly.

## 8. Simple explanation for presentation

You can explain the project like this:

> "This is an AI document assistant based on RAG. First, the application extracts text from PDF, DOCX, TXT and Markdown documents. The extracted text is divided into overlapping chunks. Sentence Transformers converts the chunks into numerical embeddings, and FAISS stores those embeddings for fast semantic search. When a user asks a question, the question is embedded and the most relevant chunks are retrieved. A simple keyword search is also performed. Both scores are combined in a hybrid search. Finally, the retrieved chunks are sent to a Groq language model, which generates an answer only from the provided document context. The retrieved filename, page number and text are displayed as sources."

## 9. Deployment on Streamlit Cloud

Upload:

```text
app.py
requirements.txt
README.md
```

to GitHub.

Then create a Streamlit app using `app.py`.

Add the secret:

```toml
RROQ_API_KEY = "your_groq_api_key_here"
```

Do not commit the API key to GitHub.

## 10. Beginner project structure

The important functions are:

```text
extract_pdf()
extract_docx()
extract_txt()
extract_md()

extract_document()

chunk_text()
create_chunks()

load_embedding_model()
build_vector_store()

keyword_search()
hybrid_search()

download_drive_file()
download_drive_folder()

process_new_document()

ask_groq()
```

This keeps the application easy to understand and extend later.

## 11. Possible future improvements

After the basic version works, you can add:

- OCR for scanned PDFs
- better DOCX page/section handling
- citations with exact source positions
- conversation history
- multiple Groq model selection
- reranking models
- persistent FAISS index on disk
- private Google Drive OAuth
- Excel/CSV support
- image/table extraction
- document summarization
- chat history
- authentication
