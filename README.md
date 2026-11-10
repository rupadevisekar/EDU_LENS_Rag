# EduLens — Education RAG System

A Streamlit application that lets students upload academic PDFs and ask questions grounded in those documents.

## Stack

- Python
- Streamlit UI
- LangChain orchestration
- Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) for local embeddings
- FAISS for vector similarity search
- Gemini API for answer generation
- PyPDF for PDF text extraction

## What is implemented

1. Upload multiple academic PDFs.
2. Extract text page by page and preserve filename/page metadata.
3. Split text into overlapping chunks.
4. Generate local 384-dimensional sentence embeddings.
5. Index the chunks in FAISS.
6. Embed student questions using the same Sentence Transformers model.
7. Retrieve the top-k relevant passages.
8. Send retrieved passages and the question to Gemini.
9. Display the answer and retrieved source passages.

## Requirements

- Python 3.10–3.12 recommended
- Internet access for installing packages, downloading the embedding model on first run, and calling Gemini
- Gemini API key from Google AI Studio: https://aistudio.google.com/apikey

## Windows quick start

1. Extract the ZIP.
2. Open the extracted `edulens_streamlit` folder in VS Code.
3. Open a terminal in that folder.
4. Run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. The app opens at the local URL printed by Streamlit (usually http://localhost:8501).
6. In the left sidebar, paste your Gemini API key.
7. Upload PDFs and click **Build knowledge base**.
8. Ask questions in the chat box.

You can also copy `.env.example` to `.env` and place your key in `GOOGLE_API_KEY`. The app also accepts a key directly in the sidebar. Do not commit or share your `.env` file.

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

`.env.example` contains:

```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-3.8-flash
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

The embedding model is downloaded and cached locally on first use. Set `EMBEDDING_DEVICE=cuda` only if PyTorch and a compatible CUDA setup are installed.

## Important limitations of this MVP

- FAISS index is held in Streamlit session state; it is not persisted after the session ends.
- Retrieval is semantic vector search only. BM25 hybrid search and reranking are not included yet.
- Scanned/image-only PDFs need OCR; this version extracts selectable PDF text.
- Source passages are shown with filenames and page numbers. These are retrieved passages, not a guarantee that every generated claim is perfectly supported. Verify important academic answers against the cited pages.
- The app sends retrieved passages to Gemini. Do not upload confidential or restricted documents unless authorized.

## Troubleshooting

**`GOOGLE_API_KEY` missing**  
Paste the key in the sidebar or create `.env` from `.env.example`.

**First document processing takes time**  
Sentence Transformers downloads model weights the first time. Later runs use the local cache.

**PDF produces no text**  
It may be scanned or image-only. OCR is a future extension.

**Gemini API error**  
Check the API key, internet connection, enabled API, quota, and model name in `.env`.

**FAISS install error on Windows**  
Use 64-bit Python 3.10–3.12 and update pip before installing requirements.

## Suggested next steps

- Persistent FAISS indexes and document collections
- BM25 + vector hybrid retrieval with Reciprocal Rank Fusion
- Cross-encoder reranking
- OCR for scanned notes
- Conversation-aware query rewriting
- Evaluation set for Recall@K, citation precision, faithfulness, and answer correctness
- User authentication and isolated per-user document stores


## Gemini model availability

The default chat model is `gemini-3.8-flash`, based on the model migration message shown by the Gemini API in the supplied screenshot. If Google changes model availability for your account, set `GEMINI_CHAT_MODEL` in `.env` to a currently supported model listed in Google AI Studio. Restart Streamlit after changing `.env`.


## API key configuration (.env only)

EduLens reads the Gemini key from the `.env` file. The web interface does not contain an API-key input.

1. Copy `.env.example` to a file named `.env` in the same folder as `app.py`.
2. Edit `.env` and set:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key
GEMINI_CHAT_MODEL=gemini-3.8-flash
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

3. Save the file and restart Streamlit. Do not put quotes around the key unless your key contains special characters that require quoting.
4. Keep `.env` private; it is excluded by `.gitignore`.

If the key is changed while Streamlit is running, restart the app so `load_dotenv()` reads the updated value.


## Configure the included .env file

This ZIP includes a `.env` file so you do not need to create one manually.

1. Open `.env` in the project folder.
2. Replace `PASTE_YOUR_GEMINI_API_KEY_HERE` with your real Gemini API key.
3. Save the file.
4. Restart Streamlit.

The application loads `.env` using a path relative to `app.py`. The API key is not entered on the webpage. The ZIP cannot contain your actual key because only you should have access to that secret.


## Fix for `ModuleNotFoundError: langchain_huggingface`

This build imports `HuggingFaceEmbeddings` from `langchain_community.embeddings`, so it does not require the separate `langchain-huggingface` package. After extracting this updated ZIP, install the requirements again using the same Python environment that runs Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Using `python -m streamlit` helps ensure Streamlit and the installed dependencies use the same Python interpreter.
