import os
import hashlib
import numpy as np
from google import genai
from google.genai import types
from rag.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    GEMINI_API_KEY
)

class EmbeddingClient:
    def __init__(self):
        self.provider = EMBEDDING_PROVIDER
        self.model = EMBEDDING_MODEL
        self.dim = EMBEDDING_DIM
        self.client = None

        if self.provider == "gemini":
            if not GEMINI_API_KEY:
                print("WARNING: GEMINI_API_KEY not found in env. Falling back to local offline embeddings.")
                self.provider = "local"
            else:
                try:
                    # Initialize the google-genai client
                    self.client = genai.Client(api_key=GEMINI_API_KEY)
                    self.dim = 768  # text-embedding-004 is 768-dim
                except Exception as e:
                    print(f"Error initializing Gemini client: {e}. Falling back to local offline embeddings.")
                    self.provider = "local"

    def get_embedding(self, text: str) -> list[float]:
        """Gets embedding for a single text string."""
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Gets embeddings for a list of text strings."""
        if not texts:
            return []

        if self.provider == "gemini" and self.client:
            try:
                # Call Gemini embedding API
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=texts,
                )
                # The response contains a list of ContentEmbedding objects
                # depending on the structure, we fetch the values
                embeddings = [emb.values for emb in response.embeddings]
                return embeddings
            except Exception as e:
                print(f"Gemini embedding API call failed: {e}. Falling back to local offline embeddings for this batch.")
                # Fall back to local below

        # Local Offline Embedding Fallback
        # We implement a deterministic hashing vectorizer (Bag of Words / Hash Trick)
        # to generate a dense vector of size self.dim.
        # This allows offline testing/running without any API key or external downloads,
        # and has basic search properties (matching words produce higher similarity).
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = text.lower().split()
            if not words:
                # Return a small random vector for empty text
                vec[0] = 1.0
                embeddings.append(vec.tolist())
                continue
                
            for word in words:
                # Hash the word to a bucket index
                h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
                idx = h % self.dim
                # Determine sign (positive or negative) to reduce collisions bias
                sign = 1 if ((h >> 8) % 2 == 0) else -1
                vec[idx] += sign
            
            # Add L2 normalization to make it a unit vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())

        return embeddings
