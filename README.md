# 🩺 Clinical Subspecialty Consult Index (RAG Agent)

An open-source clinical decision-support agent designed to assist physicians by index-searching and consulting verified medical literature (textbooks, PDFs, ebooks, and trusted web articles). 

By leveraging the **Retrieval-Augmented Generation (RAG)** pattern, this agent guarantees that all clinical responses are strictly grounded in files manually uploaded by the physician, eliminating the hallucinations commonly associated with raw LLM prompts.

---

## 🌟 Key Architecture Features
This project is engineered as a **modular blueprint**, featuring:

*   **Pluggable LLM Providers**: Toggle between native **Google Gemini API** (using its massive 2,000,000 token context window and free tier) and any **OpenAI-Compatible Endpoint** (supporting **DeepSeek API**, **OpenRouter**, or fully offline local **Ollama** models like Llama 3 or Qwen 2.5).
*   **Flexible Vector Storage**: Abstracts the database layer to easily toggle between **Local ChromaDB** (zero-configuration local SQL-backed indexing) and **Cloud Pinecone DB** (persistent, growing index synced online across multiple devices).
*   **In-Memory EPUB & PDF Parsing**: Custom light-weight ingestion pipeline extracting text and structural headers directly using pure Python without bloated or unstable third-party binary frameworks.
*   **Source Citations UI**: For clinical trust, the UI forces the LLM to output bracketed source citation tags (e.g., `[Snippet #1]`) and features interactive drop-down cards to inspect original textbook snippets and pages.

---

## 🛠️ Technology Stack
*   **Frontend UI**: [Streamlit](https://streamlit.io/) (Fast, lightweight Python dashboard framework)
*   **Orchestration & API clients**: `google-genai` and `openai`
*   **Vector Databases**: `chromadb` (local persistence) and `pinecone-client` (cloud serverless)
*   **Text Parsers**: `pypdf`, `beautifulsoup4`, `requests`
*   **Environment**: `python-dotenv`

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.9 or higher installed.
*   (Optional) [Ollama](https://ollama.com/) if you intend to run LLMs completely offline locally.

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the template `.env.example` file to create your local `.env`:
```bash
cp .env.example .env
```
Open `.env` in a text editor and customize the settings:

```env
# 1. Select active LLM provider ("gemini" or "openai_compatible")
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere

# 2. Select active database ("chroma" or "pinecone")
VECTOR_DB_TYPE=chroma
```

*For local Ollama or DeepSeek configurations, uncomment the OpenAI section in `.env` and provide your base URLs and model names.*

### 4. Running the Application
Launch the local Streamlit web server:
```bash
streamlit run app.py
```
This will automatically open the dashboard in your default browser at `http://localhost:8501`.

---

## 📂 Project Directory Structure
```
med-consultant-agent/
├── app.py                # Streamlit UI Dashboard (chat + upload console)
├── config.py             # Parses & validates environment configurations
├── verify.py             # Unit and integration test suite
├── core/
│   ├── __init__.py
│   ├── db.py             # VectorDB interface (abstracts Chroma & Pinecone)
│   ├── ingestor.py       # Pluggable parsers (PDF, EPUB, Web Scraping)
│   ├── retriever.py      # Retrieves relevant database context matching query
│   └── generator.py      # LLM orchestrator (Gemini native vs. OpenAI wrapper)
├── data/                 # Raw document cache & local Chroma DB (gitignored)
└── requirements.txt      # Project library dependencies
```

---

## 🧪 Verification & Testing
The project includes a standard test suite verifying chunking limits, directory integrity, and vector database mock connections. To execute verification:
```bash
python verify.py
```

---

## 🔒 Security & Privacy Notice
*   **No PHI (Protected Health Information)**: This tool is meant for literature-based research. Do not upload patient-identifying data.
*   **Offline Mode**: For total security and compliance in clinical settings, set `LLM_PROVIDER=openai_compatible`, configure `OPENAI_API_BASE=http://localhost:11434/v1`, and run local models using Ollama. No data will ever leave the computer.
