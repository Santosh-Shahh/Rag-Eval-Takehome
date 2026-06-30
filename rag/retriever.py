import argparse
from rag.store import LanceDBStore
from rag.embedding import EmbeddingClient
from rag.config import DEFAULT_K, SIMILARITY_THRESHOLD

class Retriever:
    def __init__(self, store: LanceDBStore = None, embedding_client: EmbeddingClient = None):
        self.store = store or LanceDBStore()
        self.embedding_client = embedding_client or EmbeddingClient()

    def retrieve(self, query: str, k: int = DEFAULT_K, filter_expr: str = None) -> list[dict]:
        """
        Embeds the query and searches the LanceDB store.
        Returns retrieved chunks.
        """
        # Embed query text
        query_vector = self.embedding_client.get_embedding(query)
        
        # Perform vector search
        results = self.store.search(query_vector, limit=k, filter_expr=filter_expr)
        
        # Calculate cosine similarity if distance is present.
        # LanceDB distance is L2 distance (squared Euclidean) or cosine distance.
        # By default LanceDB uses L2 distance unless configured.
        # Cosine distance = 1 - cosine_similarity.
        # Let's ensure we return both raw distance and computed similarity.
        for r in results:
            if "distance" in r:
                # Assuming LanceDB default Euclidean/L2 distance:
                # Let's keep the raw distance, but we can also estimate similarity or display distance.
                # In typical configurations, distance is L2: d^2.
                # If d is cosine distance, similarity is 1 - distance.
                # Let's output a normalized similarity score.
                # For L2 distance d^2, similarity can be approximated as 1 / (1 + d^2) or 1 - d^2/2.
                # If we're using normalized vectors, L2 distance d^2 = 2 - 2*cos_sim,
                # hence cos_sim = 1 - d^2/2.
                # Since our embeddings are L2 normalized (unit vectors), this holds perfectly!
                dist = r["distance"]
                similarity = 1.0 - (dist / 2.0)
                # Clamp similarity between 0.0 and 1.0
                r["similarity"] = max(0.0, min(1.0, similarity))
            else:
                r["similarity"] = 1.0

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG retriever.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve.")
    parser.add_argument("--filter", type=str, default=None, help="SQL metadata filter expression.")
    
    args = parser.parse_args()
    
    retriever = Retriever()
    results = retriever.retrieve(args.query, k=args.k, filter_expr=args.filter)
    
    print(f"\nRetrieved {len(results)} chunks for query: '{args.query}'\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] File: {r['file_name']} (Type: {r['file_type']}) | Similarity: {r['similarity']:.4f}")
        print(f"ID: {r['id']}")
        print(f"Content: {r['text'][:150]}...")
        print("-" * 60)
