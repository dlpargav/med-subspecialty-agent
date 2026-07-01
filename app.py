import os
import shutil
import streamlit as st

# Set page config as the very first command
st.set_page_config(
    page_title="Clinical Subspecialty Consult Index",
    page_icon="🩺",
    layout="wide"
)

# Import configurations & core modules
import config
from core.db import get_vector_store
from core.ingestor import ingest_file, ingest_url
from core.retriever import get_context
from core.generator import LLMGenerator

# ==============================================================================
# Premium CSS Theme Styling
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply Inter font throughout the application */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Title with smooth gradient */
    .title-gradient {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    /* Elegant subtitle description */
    .subtitle-styled {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassmorphic box for setup metadata and welcome widgets */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(8px);
    }
    
    /* Custom clinical style tags/badges */
    .status-badge {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.25);
        padding: 0.3rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .status-badge-db {
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.25);
        padding: 0.3rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
    }
    
    /* Citation container */
    .citation-container {
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Citation item layout */
    .citation-item {
        border-left: 3px solid #0ea5e9;
        background: rgba(15, 23, 42, 0.35);
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        border-top: 1px solid rgba(255,255,255,0.02);
        border-right: 1px solid rgba(255,255,255,0.02);
        border-bottom: 1px solid rgba(255,255,255,0.02);
    }
    
    .citation-header {
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.85rem;
        margin-bottom: 0.15rem;
    }
    
    .citation-snippet {
        font-size: 0.8rem;
        color: #94a3b8;
        font-style: italic;
        line-height: 1.4;
    }
    
    /* Remove padding default on sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize database store
@st.cache_resource
def get_db():
    return get_vector_store()

vector_store = get_db()

# Initialize session state for messages and system states
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# Sidebar - Ingestion and Settings
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.4rem; color: #f8fafc; font-weight: 600; margin-bottom: 1rem;'>⚙️ Clinical Console</h2>", unsafe_allow_html=True)
    
    # Provider display indicators
    st.markdown("<div class='glass-card' style='padding: 1rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.markdown(f"**LLM Backend**<br><span class='status-badge'>{config.LLM_PROVIDER.upper()}</span>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"**Vector DB**<br><span class='status-badge-db'>{config.VECTOR_DB_TYPE.upper()}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Literature count indicator
    try:
        total_chunks = vector_store.get_count()
    except Exception:
        total_chunks = 0
        
    st.metric(label="📚 Total Indexed Text Chunks", value=f"{total_chunks:,}")
    st.markdown("---")
    
    # Section: Upload Files
    st.markdown("<h3 style='font-size: 1.1rem; color: #f8fafc; font-weight: 600;'>📤 Feed Literature</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Medical Guidelines (PDF, EPUB, TXT)", 
        type=["pdf", "txt", "epub"], 
        accept_multiple_files=True,
        key="file_uploader_id"
    )
    
    # Section: Scrape website URL
    url_input = st.text_input("Index Online Resource (URL):", placeholder="https://www.who.int/news/...", key="url_input_id")
    
    # Processing action button
    if st.button("Index Selected Resources", key="submit_ingestion_btn_id", type="primary", use_container_width=True):
        files_indexed = 0
        urls_indexed = 0
        
        # 1. Process files
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Parsing {file.name}..."):
                    # Save to raw files folder
                    save_path = config.RAW_DOCS_DIR / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Run ingestion pipeline
                    chunks_added = ingest_file(str(save_path), vector_store)
                    if chunks_added > 0:
                        files_indexed += 1
                        
        # 2. Process URL
        if url_input and url_input.strip():
            with st.spinner(f"Scraping web resource..."):
                chunks_added = ingest_url(url_input.strip(), vector_store)
                if chunks_added > 0:
                    urls_indexed += 1
                    
        # Summarize results
        if files_indexed > 0 or urls_indexed > 0:
            st.success(f"Success! Indexed {files_indexed} file(s) and {urls_indexed} URL(s).")
            # Clear file uploader cache state by refreshing
            st.rerun()
        else:
            st.warning("No new content was indexed. Verify files or URL is valid.")

# ==============================================================================
# Main Layout - Chat Interface
# ==============================================================================
st.markdown("<h1 class='title-gradient'>🩺 Clinical Subspecialty Consult Index</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-styled'>Grounded Clinical Decision-Support Assistant (RAG Pattern)</p>", unsafe_allow_html=True)

# Welcome user if DB has no documents
if total_chunks == 0:
    st.markdown(
        """
        <div class='glass-card'>
            <h3 style='margin-top: 0; color: #f8fafc; font-weight: 600;'>Welcome to your Clinical Assistant!</h3>
            <p style='color: #94a3b8; line-height: 1.6;'>
                Your knowledge index is currently empty. To start querying the agent:
            </p>
            <ol style='color: #e2e8f0; line-height: 1.8;'>
                <li>Upload verified literature (e.g. medical textbooks, subspecialty guidelines) using the sidebar.</li>
                <li>Enter links to trusty medical resources or clinical trials.</li>
                <li>Click <strong>Index Selected Resources</strong>.</li>
            </ol>
            <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;'>
                Once indexed, the agent will base all clinical queries exclusively on those verified sources.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If assistant response has citations, display them inside an expander
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📚 View Supporting Literature Citations"):
                    st.markdown("<div class='citation-container'>", unsafe_allow_html=True)
                    for cite in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class='citation-item'>
                                <div class='citation-header'>[{cite['index']}] {cite['label']}</div>
                                <div class='citation-snippet'>"{cite['snippet']}"</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input Box
    if user_query := st.chat_input("Ask a clinical/subspecialty question...", key="chat_input_id"):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # 2. Query DB and Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving literature context and generating clinical response..."):
                # Retrieve matching snippets
                context_str, citations = get_context(user_query, k=5)
                
                # Generate LLM response
                try:
                    generator = LLMGenerator()
                    response_text = generator.generate_response(user_query, context_str)
                except Exception as e:
                    response_text = f"Initialization Error: {str(e)}"
                    citations = []
                
                # Render Response text
                st.markdown(response_text)
                
                # Render citations list if available
                if citations:
                    with st.expander("📚 View Supporting Literature Citations"):
                        st.markdown("<div class='citation-container'>", unsafe_allow_html=True)
                        for cite in citations:
                            st.markdown(
                                f"""
                                <div class='citation-item'>
                                    <div class='citation-header'>[{cite['index']}] {cite['label']}</div>
                                    <div class='citation-snippet'>"{cite['snippet']}"</div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        st.markdown("</div>", unsafe_allow_html=True)
                        
            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "citations": citations
            })
