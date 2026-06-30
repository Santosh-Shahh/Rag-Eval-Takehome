import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", str(DATA_DIR / "lancedb"))

# Embedding Configurations
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # "local" or "gemini"
# For local embedding, we use sentence-transformers. 
# We'll write a simple fallback to a pure-python or sklearn tf-idf or sentence-transformers model.
# Let's support sentence-transformers model if installed, else fallback to a simple mock or tf-idf/numpy-based.
# This ensures zero-config running!
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384")) # all-MiniLM-L6-v2 is 384, text-embedding-004 is 768

# LLM Configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "mock"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Ingestion Configurations
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))  # Cosine similarity threshold for RAG

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
