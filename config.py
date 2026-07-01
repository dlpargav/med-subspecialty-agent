import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "vector_db"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"

# Create directories if they do not exist
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
RAW_DOCS_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# OpenAI Compatible configuration
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "llama3")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "nomic-embed-text")

# Database Configuration
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma").lower()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "med-consultant")

def validate_config():
    """
    Validates that the required environment variables are set.
    Prints informative messages or warnings to help the user.
    """
    warnings = []
    
    # Validate LLM configs
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            warnings.append("Warning: GEMINI_API_KEY is not set. Gemini API calls will fail.")
    elif LLM_PROVIDER == "openai_compatible":
        if not OPENAI_API_KEY and "localhost" not in OPENAI_API_BASE:
            warnings.append("Warning: OPENAI_API_KEY is not set for a remote OpenAI-compatible endpoint.")
    else:
        warnings.append(f"Error: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Must be 'gemini' or 'openai_compatible'.")
        
    # Validate DB configs
    if VECTOR_DB_TYPE == "pinecone":
        if not PINECONE_API_KEY:
            warnings.append("Warning: PINECONE_API_KEY is not set. Pinecone database connection will fail.")
        if not PINECONE_INDEX_NAME:
            warnings.append("Warning: PINECONE_INDEX_NAME is not set.")
    elif VECTOR_DB_TYPE != "chroma":
        warnings.append(f"Error: Unknown VECTOR_DB_TYPE '{VECTOR_DB_TYPE}'. Must be 'chroma' or 'pinecone'.")
        
    if warnings:
        print("\n=== Configuration Verification ===")
        for warning in warnings:
            print(warning)
        print("==================================\n")
    else:
        print("Configuration successfully validated.")

# Run validation on load
validate_config()
