import os
import json
import time
import re
import numpy as np
from pathlib import Path
from google import genai

from rag.config import GEMINI_API_KEY, LLM_PROVIDER
from rag.store import LanceDBStore
from rag.embedding import EmbeddingClient
from rag.retriever import Retriever
from rag.generator import Generator

def compute_f1(prediction, ground_truth):
    """Computes F1 score based on token overlap."""
    pred_tokens = re.sub(r'[^\w\s]', '', prediction.lower()).split()
    gt_tokens = re.sub(r'[^\w\s]', '', ground_truth.lower()).split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_em(prediction, ground_truth):
    """Computes Exact Match (EM) score."""
    pred_clean = re.sub(r'[^\w\s]', '', prediction.lower()).strip()
    gt_clean = re.sub(r'[^\w\s]', '', ground_truth.lower()).strip()
    return 1.0 if pred_clean == gt_clean else 0.0

class RAGEvaluator:
    def __init__(self):
        self.store = LanceDBStore()
        self.emb_client = EmbeddingClient()
        self.retriever = Retriever(self.store, self.emb_client)
        self.generator = Generator()
        self.judge_client = None

        if GEMINI_API_KEY:
            try:
                self.judge_client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                print(f"Error initializing judge client: {e}")

    def judge_llm(self, query: str, context: str, answer: str, metric: str) -> float:
        """
        Uses an LLM Judge to evaluate Faithfulness or Relevance on a 1-5 scale.
        If offline, uses heuristic word overlap/length comparisons.
        """
        if not self.judge_client:
            # Local Heuristic Fallback
            if metric == "faithfulness":
                # Check how many words from the answer are in the context
                ans_words = set(re.sub(r'[^\w\s]', '', answer.lower()).split())
                ctx_words = set(re.sub(r'[^\w\s]', '', context.lower()).split())
                if not ans_words:
                    return 5.0
                overlap = len(ans_words & ctx_words) / len(ans_words)
                # Map overlap to 1-5 scale
                return round(1.0 + 4.0 * overlap, 1)
            else: # relevance
                # Check keyword overlap between answer and query
                q_words = set(re.sub(r'[^\w\s]', '', query.lower()).split()) - {"what", "is", "the", "are", "of", "in", "for", "a", "an", "does"}
                ans_words = set(re.sub(r'[^\w\s]', '', answer.lower()).split())
                if not q_words:
                    return 5.0
                overlap = len(q_words & ans_words) / len(q_words)
                return round(1.0 + 4.0 * overlap, 1)

        # Gemini LLM Judge Prompts
        prompts = {
            "faithfulness": (
                f"You are an LLM Judge evaluating RAG answers.\n"
                f"Assess if the generated answer is strictly supported by the context below, without hallucination.\n"
                f"Score from 1 (completely hallucinated/unsupported) to 5 (fully faithful and supported by context).\n\n"
                f"Context: {context}\n"
                f"Answer: {answer}\n\n"
                f"Return ONLY a JSON object: {{\"score\": float, \"rationale\": \"string\"}}"
            ),
            "relevance": (
                f"You are an LLM Judge evaluating RAG answers.\n"
                f"Assess if the generated answer directly addresses and answers the user query.\n"
                f"Score from 1 (irrelevant/off-topic) to 5 (perfectly relevant and directly answers the query).\n\n"
                f"Query: {query}\n"
                f"Answer: {answer}\n\n"
                f"Return ONLY a JSON object: {{\"score\": float, \"rationale\": \"string\"}}"
            )
        }

        try:
            response = self.judge_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompts[metric]
            )
            # Find JSON block
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return float(data.get("score", 3.0))
            return 3.0
        except Exception as e:
            print(f"Error in LLM Judge call: {e}")
            return 3.0

    def run_eval(self, qa_data_path: Path, k: int = 4) -> dict:
        with open(qa_data_path, "r") as f:
            qa_pairs = json.load(f)

        print(f"Running evaluation on {len(qa_pairs)} questions...")
        
        results = []
        retrieval_latencies = []
        end_to_end_latencies = []

        recall_hits = 0
        reciprocal_ranks = []
        ndcgs = []
        context_precisions = []
        
        faithfulness_scores = []
        relevance_scores = []
        ems = []
        f1s = []

        for idx, item in enumerate(qa_pairs):
            query = item["question"]
            gold_file = item["gold_file"]
            gold_answer = item["gold_answer"]

            # 1. Retrieve
            start_retrieval = time.time()
            retrieved = self.retriever.retrieve(query, k=k)
            ret_latency = (time.time() - start_retrieval) * 1000
            retrieval_latencies.append(ret_latency)

            # Calculate Retrieval Metrics
            # Target relevant file: gold_file
            # Check ranks
            hit = False
            first_rank = 0
            rel_list = []
            
            for rank_idx, chunk in enumerate(retrieved):
                chunk_file = chunk["file_name"]
                is_rel = 1 if chunk_file.lower() == gold_file.lower() else 0
                rel_list.append(is_rel)
                
                if is_rel == 1 and not hit:
                    hit = True
                    first_rank = rank_idx + 1

            # Hit Rate / Recall@k
            if hit:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            # nDCG@k
            dcg = 0.0
            for rank_idx, rel in enumerate(rel_list):
                dcg += rel / np.log2(rank_idx + 2)
            ndcgs.append(dcg) # since IDCG = 1.0 (ideal puts 1 rel doc at rank 1)

            # Context Precision
            num_rel = sum(rel_list)
            if num_rel == 0:
                context_precision = 0.0
            else:
                precisions = []
                rel_count = 0
                for r_idx, rel in enumerate(rel_list):
                    if rel == 1:
                        rel_count += 1
                        precision_at_i = rel_count / (r_idx + 1)
                        precisions.append(precision_at_i)
                context_precision = sum(precisions) / num_rel
            context_precisions.append(context_precision)

            # 2. Generate
            start_e2e = time.time()
            gen_out = self.generator.generate_answer(query, retrieved)
            e2e_latency = (time.time() - start_e2e) * 1000
            end_to_end_latencies.append(e2e_latency)

            answer = gen_out["answer"]
            
            # Answer quality
            em = compute_em(answer, gold_answer)
            f1 = compute_f1(answer, gold_answer)
            ems.append(em)
            f1s.append(f1)

            # Context for judge (combine retrieved chunks text)
            context_text = "\n\n".join([c["text"] for c in retrieved])
            
            # LLM Judge Faithfulness & Relevance
            faithfulness = self.judge_llm(query, context_text, answer, "faithfulness")
            relevance = self.judge_llm(query, context_text, answer, "relevance")
            faithfulness_scores.append(faithfulness)
            relevance_scores.append(relevance)

            results.append({
                "id": item["id"],
                "question": query,
                "gold_file": gold_file,
                "gold_answer": gold_answer,
                "retrieved_files": [c["file_name"] for c in retrieved],
                "retrieved_scores": [c.get("similarity", 1.0) for c in retrieved],
                "generated_answer": answer,
                "hit": hit,
                "rank": first_rank if hit else -1,
                "ndcg": dcg,
                "context_precision": context_precision,
                "em": em,
                "f1": f1,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "retrieval_latency_ms": ret_latency,
                "e2e_latency_ms": e2e_latency
            })

            print(f"Eval [{idx+1}/{len(qa_pairs)}]: Hit={hit} | F1={f1:.4f} | Faith={faithfulness:.1f} | E2E Latency={e2e_latency:.0f}ms")

        # Aggregate Stats
        total_q = len(qa_pairs)
        metrics = {
            "retrieval": {
                "recall_at_k": recall_hits / total_q,
                "mrr": np.mean(reciprocal_ranks),
                "ndcg_at_k": np.mean(ndcgs),
                "context_precision": np.mean(context_precisions)
            },
            "generation": {
                "faithfulness": np.mean(faithfulness_scores),
                "relevance": np.mean(relevance_scores),
                "em": np.mean(ems),
                "f1": np.mean(f1s)
            },
            "latency": {
                "retrieval_p50": np.percentile(retrieval_latencies, 50),
                "retrieval_p95": np.percentile(retrieval_latencies, 95),
                "e2e_p50": np.percentile(end_to_end_latencies, 50),
                "e2e_p95": np.percentile(end_to_end_latencies, 95)
            }
        }

        # Save to disk
        out_results_path = qa_data_path.parent / "eval_results.json"
        with open(out_results_path, "w") as f:
            json.dump({"metrics": metrics, "detailed_runs": results}, f, indent=2)

        print("\n=== EVALUATION REPORT ===")
        print(f"Total Questions Evaluated: {total_q}")
        print("-" * 30)
        print("Retrieval Metrics:")
        print(f"  Recall@{k} (Hit Rate): {metrics['retrieval']['recall_at_k']:.4f}")
        print(f"  MRR:                 {metrics['retrieval']['mrr']:.4f}")
        print(f"  nDCG@{k}:             {metrics['retrieval']['ndcg_at_k']:.4f}")
        print(f"  Context Precision:   {metrics['retrieval']['context_precision']:.4f}")
        print("Generation Quality:")
        print(f"  Faithfulness:        {metrics['generation']['faithfulness']:.2f}/5.0")
        print(f"  Answer Relevance:    {metrics['generation']['relevance']:.2f}/5.0")
        print(f"  Exact Match (EM):    {metrics['generation']['em']:.4f}")
        print(f"  F1 Score:            {metrics['generation']['f1']:.4f}")
        print("Latency (ms):")
        print(f"  Retrieval p50:       {metrics['latency']['retrieval_p50']:.1f}ms")
        print(f"  Retrieval p95:       {metrics['latency']['retrieval_p95']:.1f}ms")
        print(f"  End-to-End p95:      {metrics['latency']['e2e_p95']:.1f}ms")
        print("=========================\n")

        return metrics

if __name__ == "__main__":
    qa_path = Path(__file__).resolve().parent.parent / "data" / "qa_eval.json"
    
    # Run ingestion first to ensure vector DB has up to date content
    store = LanceDBStore()
    if store.count_vectors() == 0:
        print("Vector store is empty. Triggering scanning and ingestion of corpus directory...")
        emb_client = EmbeddingClient()
        from rag.ingest import ingest_directory
        ingest_directory(Path(__file__).resolve().parent.parent / "data" / "corpus", store, emb_client)

    evaluator = RAGEvaluator()
    evaluator.run_eval(qa_path)
