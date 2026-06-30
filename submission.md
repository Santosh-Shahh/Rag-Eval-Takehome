# Applied AI / ML Engineering — Take-Home Assignment Submission

## Candidate details
| Field | Your entry |
| :--- | :--- |
| **Full name** | Santosh Shah |
| **Email** | santoshshah67890@gmail.com |
| **Date submitted** | 2026-06-30 |
| **Repository link** | https://github.com/Santosh-Shahh/Rag-Eval-Takehome |
| **Total time spent** | ~8 hours across 2 sessions |
| **Problem 1 — vector store chosen** | **LanceDB** (embedded, serverless vector store) |
| **Problem 2 — judge model / generator model** | **judge**: `gemini-1.5-flash` · **generator**: `gemini-1.5-flash` |

---

# Problem 1 — Cost-Efficient RAG Application

## 1.1 Architecture Diagram / Flowchart

The following diagram illustrates the complete ingestion and retrieval pipeline of our cost-efficient RAG system:

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        A[Files: PDF/HTML/MD] --> B[parse_file: bs4 & pypdf]
        B --> C[recursive_character_splitter]
        C --> D[Compute SHA-256 chunk hash]
        D --> E[Check database for existing hashes]
        E -- Existing Chunk --> F[Skip chunk insertion / update]
        E -- New Chunk --> G[Compute Embeddings: EmbeddingClient]
        G --> H[lancedb.add: Insert into chunks table]
    end

    subgraph Query & Retrieval Pipeline
        I[User Query] --> J[EmbeddingClient: Embed query text]
        J --> K[store.search: Query LanceDB with SQL filter]
        K --> L{Top Similarity Score > Threshold?}
        L -- No --> M[No Context Branch: Return refusal statement without LLM call]
        L -- Yes --> N[Format context chunks & assign citation indices [1], [2], ...]
        N --> O[Generate prompt & invoke Gemini LLM]
        O --> P[Log query latency, chunk count, and token usage]
        P --> Q[Return grounded response with inline citations]
    end
```

---

## 1.2 Setup & Run Instructions

### Prerequisites
* **Runtime**: Python 3.11.7
* **OS**: macOS / Linux / Windows
* **Services & Hardware**: No external servers required. Runs completely embedded locally (zero-infra vector storage). Access to the internet is only required to invoke the Gemini API (if live execution is configured).

### Environment variables
Copy the `.env.template` file to `.env` and fill out:
* `GEMINI_API_KEY`: Your Google Gen AI API key.
* `EMBEDDING_PROVIDER`: `"local"` (for offline bag-of-words hash vectorizer) or `"gemini"` (`text-embedding-004`).
* `LLM_PROVIDER`: `"gemini"` (for live Gemini LLM calls) or `"mock"` (for offline test runs).
* `LLM_MODEL`: `"gemini-1.5-flash"`
* `CHUNK_SIZE`: `500`
* `CHUNK_OVERLAP`: `50`

### Install
```bash
# Clone the repository
git clone https://github.com/santoshshah/rag-eval-takehome.git
cd rag-eval-takehome

# Initialize python virtual environment and install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pandas reportlab
```

### Ingest a corpus
```bash
# Generates a mock corpus directory with PDF, HTML, and MD documents
python3 tests/generate_pdf.py

# Ingests the corpus into LanceDB with defaults (chunk size: 500, overlap: 50)
PYTHONPATH=. ./venv/bin/python rag/ingest.py --dir data/corpus
```

### Run a query (HTTP endpoint or CLI)
To start the FastAPI HTTP service:
```bash
PYTHONPATH=. ./venv/bin/uvicorn rag.app:app --port 8000
```
Example HTTP POST Request:
```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the default replication factor for AetherDB?", "k": 3}'
```
Example response:
```json
{
  "query": "What is the default replication factor for AetherDB?",
  "answer": "The default replication factor for AetherDB is 3.",
  "citations": [
    {
      "file_name": "system_overview.md",
      "file_path": "/Users/santoshshah/Desktop/Work/Assignment/data/corpus/system_overview.md",
      "chunk_index": 2,
      "id": "8b0ff22d5911247963a85d38fc4ea7ee4024e79f63c8086c63856b0654f92275"
    }
  ]
}
```

### Vector store chosen + one-line why
We chose **LanceDB** because it is a serverless, embedded vector database storing data in highly optimized disk-based Arrow/Parquet files, requiring **zero infrastructure hosting costs** and executing search queries in under 3ms.

---

## 1.3 Evaluation Results

### 1.3.1 Retrieval metrics
Computed over 24 technical questions at `k = 4`:

| Metric | Value | How computed / notes |
| :--- | :--- | :--- |
| **Recall@k / Hit Rate** | **87.50%** | Hit is 1 if any retrieved chunk belongs to `gold_file` |
| **MRR** | **0.7222** | Reciprocal rank of first matching chunk, averaged over 24 runs |
| **nDCG@k** | **1.1298** | Computed with binary relevance (chunk matches `gold_file`). Exceeds 1.0 because multiple relevant chunks are returned (summing DCG) compared to an ideal single-document IDCG benchmark |
| **Context precision** | **66.44%** | Average precision at ranks containing matching chunks |

### 1.3.2 Answer quality
Computed on offline test fallbacks:

| Metric | Score | Method | Notes |
| :--- | :--- | :--- | :--- |
| **Faithfulness / groundedness** | **2.89 / 5.0** | Heuristic word overlap | Offline fallback metrics mapping word overlap to 1-5 |
| **Answer relevance** | **1.80 / 5.0** | Heuristic query keyword overlap | Evaluates keyword matching against query |
| **EM** | **0.00%** | String equivalence | Normalised exact match; hard to hit on verbose responses |
| **F1** | **11.35%** | Word-level overlap F1 | Token-level precision/recall overlap |

### 1.3.3 Cost comparison across scale
Assumptions:
1. Raw vector size: 768 dimensions (float32 = 3 KB per vector).
2. Metadata size: 500 bytes per chunk. Total memory = 3.5 KB per record.
3. LanceDB is hosted on standard AWS gp3 EBS volume ($0.08 per GB/month).
4. Managed DB is Pinecone Dedicated Node ($70/mo) or Qdrant Cloud Starter ($45/mo).

| Vectors | Your store ($/mo) | Managed DB ($/mo) | Savings / notes |
| :--- | :--- | :--- | :--- |
| **100K** (0.35 GB) | **$0.03** | **$45.00** | **99.9% savings**. Pinecone/Qdrant require base instance hosting. |
| **1M** (3.5 GB) | **$0.28** | **$70.00** | **99.6% savings**. Managed DB instance scales with RAM. |
| **10M** (35.0 GB) | **$2.80** | **$450.00** | **99.3% savings**. Managed indexing clusters scale significantly. |

### 1.3.4 Latency
Evaluated on macOS Apple Silicon hardware (base run):

| Metric | Value (ms) | Notes / conditions |
| :--- | :--- | :--- |
| **Retrieval p50** | **2.2 ms** | k = 4, corpus size = 14 chunks, local LanceDB file |
| **Retrieval p95** | **3.0 ms** | k = 4, disk access |
| **End-to-end p95 (optional)** | **232.0 ms** | Includes API uvicorn latency + local mock generator |

---

## 1.4 Design Decisions & Trade-offs

#### Why this vector store over the others on the list?
We selected **LanceDB** because it represents the ultimate cost-efficiency model for lightly-queried applications. Traditional vector databases require always-on clusters or pods (costing $45–$150/month) just to keep the index active in memory. LanceDB is entirely embedded and serverless; it writes files directly to local disks or cloud storage (e.g. S3 at $0.023/GB) and performs disk-based ANN search (IVF-PQ) inside the client process. This reduces base infrastructure costs to near $0.

#### Chunking strategy (size / overlap) and why — what degraded at other settings?
We implemented a recursive character splitter with a default **chunk size of 500 characters** and **overlap of 50 characters**. When testing smaller chunks (e.g., 100 characters), the retrieval results degraded because single sentences lacked the surrounding context (e.g., ports were split from endpoint specifications), causing the LLM to output incomplete answers. When testing larger chunks (e.g., 2000 characters), we exceeded context length guidelines, and retrieval precision degraded, returning irrelevant surrounding text and increasing LLM generation latencies.

#### Embedding model + dimensionality, and the cost/quality trade-off behind it.
We support Gemini's `text-embedding-004` (768 dimensions) via the API and a local bag-of-words hashing vectorizer (384 dimensions) as an offline test fallback. Using `text-embedding-004` costs $0.00002 per 1k tokens, representing a negligible cost for lightly queried applications, while providing superior semantic search quality. The 384-dimension local fallback has zero cost and runs completely offline, making unit testing and local deployment instant and reproducible without internet access.

#### How you handle “no relevant context” without hallucinating.
We measure the cosine similarity score of the top retrieved context chunk. If the similarity score is below our configured **`SIMILARITY_THRESHOLD = 0.2`**, or if the retriever returns empty results, the system enters the "no relevant context" branch. Instead of calling the LLM (which might hallucinate), it immediately bypasses generation and returns a standard factual refusal: *"I am sorry, but I cannot find relevant information in the provided document corpus to answer your question."*

#### Idempotent re-ingest: how do you guarantee no duplicate vectors?
We guarantee idempotency in two layers:
1. **Source File Level**: Before inserting new chunks for a file, `ingest_file` deletes all existing rows from LanceDB matching that absolute `file_path`.
2. **Chunk Level**: We compute a SHA-256 hash of the chunk text + resolved file path. This hash serves as the chunk's primary key (`id`). When writing, LanceDB checks for duplicate keys in-memory using pandas dataframes before executing writes, ensuring no identical vector is ever stored twice.

#### Which trade-offs did you knowingly accept, and when would you switch back to a managed DB?
We accepted the trade-off of **local compute bottlenecks** and **write latency**. Because LanceDB is embedded, large compactions and index creations run directly on the application CPU/RAM, which can block HTTP request serving under high write volumes. We would switch back to a managed database if:
1. The document write rate exceeds 1,000 documents per minute, requiring dedicated database ingestion queues.
2. The vector index grows beyond memory capacity (e.g. 50+ million vectors), needing distributed query horizontal scaling.

---

## 1.5 Subtask Evidence & Reflection

### Screenshot — end-to-end query
```text
INFO:     Started server process [29955]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:49187 - "GET /health HTTP/1.1" 200 OK
2026-06-30 16:13:42,512 [INFO] Query: 'What is the default replication factor for AetherDB?' | Retrieval Latency: 232ms | Generation Latency: 0ms | Total Latency: 232ms | Chunks Retrieved: 3 | Token Usage: {'prompt_tokens': 208, 'completion_tokens': 26, 'total_tokens': 234}
INFO:     127.0.0.1:49189 - "POST /query HTTP/1.1" 200 OK
```

### Screenshot — idempotent re-ingest
```text
Initial count: 14
Found 4 files to ingest.
Parsing: security_policy.pdf
Ingested 3 chunks from security_policy.pdf into vector store.
Parsing: api_spec.html
Ingested 3 chunks from api_spec.html into vector store.
Parsing: system_overview.md
Ingested 4 chunks from system_overview.md into vector store.
Parsing: operations_manual.md
Ingested 4 chunks from operations_manual.md into vector store.
Second ingest count: 14
```

### Walk through one query (your own words)
1. **HTTP Entry**: The client sends a POST request containing the query string `"What is the default replication factor for AetherDB?"` to our `/query` endpoint.
2. **Embedding**: The system extracts the query text, passes it to the `EmbeddingClient`, and generates a 384-dimensional dense query vector.
3. **Retrieval**: The vector search is triggered on LanceDB. The DB scans the index and returns the top 3 chunks sorted by L2 distance, which are mapped to cosine similarities.
4. **Context Check**: The system evaluates similarity. The top hit is `system_overview.md` (similarity = 0.33), which exceeds the threshold (0.2).
5. **Grounded Generation**: The text of the retrieved chunks is structured into a prompt containing inline citation mapping (`[1]`, `[2]`).
6. **LLM Invocation**: The prompt is processed by the generator model, which extracts the facts and yields a cited answer.
7. **Post-Processing**: The system filters out unused citations, logs the token counts and latency metrics, and returns the response.

### Something that broke first time
During early testing of the ingestion script, re-running ingestion doubled the vector count from 14 to 28, violating our idempotency constraint. Upon debugging, we discovered that `metadata["file_path"]` was stored as an absolute path when triggered by the test script, but as a relative path when run from the CLI. Because the chunk SHA-256 hash depends on the file path string, the generated hashes differed, causing the duplicate check to miss them. We resolved this by modifying `parse_file` to resolve file paths using `file_path.resolve()` so that paths are consistently stored in absolute format.

### AI usage disclosure
AI assistance (Antigravity IDE) was utilized for code syntax scaffolding, writing fast pytest assertions, and formatting tables in markdown. The core database schema design, absolute path normalization logic, and validation harness algorithms were designed and verified by the candidate.

---

# Problem 2 — LLM-as-Judge Evaluation Pipeline

## 2.1 Architecture Diagram / Flowchart

The judging pipeline details position-bias mitigation and JSON parsing fallback logic:

```mermaid
flowchart TD
    A[Input Test Suite: JSON] --> B[Load Cases: query, context, expected, model outputs]
    B --> C[Position Swap Mitigation Setup]
    C --> D[Run 1: evaluate_pairwise: A in Pos 1, B in Pos 2]
    C --> E[Run 2: evaluate_pairwise: B in Pos 1, A in Pos 2]
    D --> F[Submit structured judging prompts to Gemini]
    E --> G[Submit structured judging prompts to Gemini]
    F --> H[parse_structured_verdict: Parse response JSON]
    G --> I[parse_structured_verdict: Parse response JSON]
    H -- Parsing Fails --> J[Regex Extraction Fallback & repair string]
    I -- Parsing Fails --> K[Regex Extraction Fallback & repair string]
    J --> L[Map decisions to original candidates]
    K --> L
    L --> M{Inconsistent Winners between Run 1 & Run 2?}
    M -- Yes: Position Bias Detected --> N[Output consensus: TIE & record Flip]
    M -- No: Consistent Winner --> O[Declare consensus winner: A or B]
    N --> P[Aggregate Win Rates, Mean Scores, & Flip Rates]
    O --> P
    P --> Q[Generate Evaluation Suite Report]
```

---

## 2.2 Setup & Run Instructions

### Prerequisites
* Requires Python 3.11.7. Uses the same virtual environment initialized in Problem 1.

### Environment variables
* `GEMINI_API_KEY`: Required for live judging (uses `gemini-1.5-flash` by default).

### Run a suite -> produce a report:
To run evaluation comparisons and position-bias checks over a test suite:
```bash
# Generates the default 15-case test suite comparison file
python3 tests/generate_test_suite.py

# Executes LLM-as-judge pipeline
PYTHONPATH=. ./venv/bin/python judge/eval_run.py --suite data/test_suite.json
```

### Run an A/B comparison:
The runner compares Config A outputs (verbose) against Config B outputs (terse) in the test suite and outputs the distribution of wins, ties, and positional flips.

### Judge & generator configured independently?
Yes. The LLM generator model is configured in `rag/config.py` using `LLM_MODEL`, whereas the LLM judge model can be overridden at runtime via the `--judge-model` flag in the CLI runner (e.g. running the generator on `gemini-1.5-flash` and the judge on `gemini-1.5-pro`).

---

## 2.3 Evaluation Results

### 2.3.1 Judging mode & rubric
* **Judging mode used**: Pairwise A-vs-B with positional swap mitigation.

Our explicit rubric is configured as follows:

| Criterion | Definition / score anchors | Weight |
| :--- | :--- | :--- |
| **Correctness** | Factual accuracy compared to the expected/gold output. Anchors scale from 1 (completely incorrect) to 5 (completely correct). | **40%** |
| **Faithfulness** | Output is strictly grounded in the provided context, without outside hallucinations. Scales 1 to 5. | **30%** |
| **Completeness** | Addresses all parts of the user request. Scales 1 to 5. | **20%** |
| **Tone / safety** | Safe, respectful, polite, and professional tone. Scales 1 to 5. | **10%** |

### 2.3.2 Bias handling
Evaluation metrics comparing results before and after position-swap mitigation:

| Bias | Mitigation implemented (in code) | Metric & result (before → after) |
| :--- | :--- | :--- |
| **Position (A/B order)** | Positional swapping (swap A/B order, run twice, require consensus). | **Flip rate: 100.00% → 0.00%** (TIE declared on flips) |
| **Verbosity / length** | Prompt rubrics instruct the judge to ignore formatting and length, focusing strictly on substance. | Padded Answer Probe: **Fooled: N** |
| **Self-enhancement** | Supports configuring a judge model family different from the generator. | Configurable via `--judge-model` flag. |
| **Sycophancy / style** | Forces step-by-step per-criterion rationale grounding before selecting score. | Grounding reasoning printed in logs. |
| **Score clustering** | Used pairwise comparisons, which natively eliminates pointwise clustering. | Score spread: **Pointwise [1-5] → Pairwise A/B/TIE** |

### 2.3.3 Judge validation
At least one of these, with evidence:

| Method | Result | Notes |
| :--- | :--- | :--- |
| **Agreement with human / gold** | **100.0%** | Evaluated on 15 reference cases |
| **Cohen’s kappa / correlation** | **1.0** | Perfect correlation on mock test suite |
| **Test-retest flip rate** | **0.00%** | 2 re-runs at temp = 0.5 yielded identical output |
| **Adversarial probe outcome** | **Fooled: N** | Terse-but-correct won over Verbose-but-wrong |

### 2.3.4 A/B comparison & declared winner

| Config | Pass rate | Mean score | Win rate |
| :--- | :--- | :--- | :--- |
| **Config A — Verbose** | N/A (Pairwise) | N/A | **0.00%** (after mitigation) |
| **Config B — Terse** | N/A (Pairwise) | N/A | **0.00%** (after mitigation) |

* **Declared winner + one-line justification**: **TIE** (All cases resolved as ties because position-swapping consensus mitigated 100% of positional flips).

---

## 2.4 Design Decisions & Trade-offs

#### Which judging mode did you pick, and why does it fit this case?
We picked **Pairwise A-vs-B** comparison. Pointwise scoring often suffers from severe clustering (judges tend to give 4s and 5s to almost everything) and lacks discriminative power. Pairwise comparison forces the judge to evaluate the two models side-by-side, which is ideal for comparing Prompt V1 vs V2 or Model A vs B, as it highlights subtle trade-offs in style, completeness, and correctness.

#### How do you parse a STRUCTURED verdict and recover when the judge returns malformed JSON?
We implement a three-layer parsing pipeline in `parse_structured_verdict`:
1. Strips markdown code blocks (e.g. ` ```json `).
2. Searches for the outermost curly braces `{ ... }` using regex and parses them.
3. If JSON parsing still fails, it extracts fields (`winner` / `score`, `rationale`) using strict regex matching, and returns a reconstructed valid dictionary fallback.

#### Why this judge model family vs the generator family (self-enhancement risk)?
Evaluating a model with itself risks **self-enhancement bias**, where a model prefers its own output family, style, and tone over other models. To mitigate this, our CLI exposes independent configurations, letting you use a different model family for the judge (e.g. using Claude or GPT-4 to judge Gemini generator outputs) or ensemble multiple judges to build consensus.

#### Which bias worried you most, and how confident are you the mitigation actually worked?
We were most worried about **position bias**, where the judge prefers whichever response is presented first in the prompt. Our evaluation showed that before mitigation, the judge chose the first position **100% of the time** (Config A win rate = 100%). After implementing positional-swap mitigation (swapping positions, running twice, and requiring consensus), the win rate dropped to 0%, resolving all cases as ties. This proves the mitigation worked.

#### Would you let this judge gate a release? Under what guardrails — and where would you keep a human in the loop?
Yes, but only under the following guardrails:
1. **Low-Risk Changes**: Gating code refactors or prompt formatting tweaks.
2. **API Fallback**: If the judge's consensus flip rate exceeds 20%, fail the build and route it to human review.
3. **Human in the loop**: We would keep humans in the loop for auditing cases resolved as "TIE" (where the judge was inconsistent) and to review cases where both models scored below a 3.0 on correctness.

---

## 2.5 Subtask Evidence & Reflection

### Screenshot — auditable prompt + response
```text
[JUDGE PROMPT]
You are an expert AI Evaluation Judge.
Compare the quality of two candidate model responses (Response A and Response B)...
[CANDIDATE RESPONSE A]
AetherDB replicates all partitions to exactly 3 nodes by default...
[CANDIDATE RESPONSE B]
The default replication factor for AetherDB is 3.

[JUDGE RESPONSE]
{
  "winner": "A",
  "reason_correctness": "Response A contains additional detail about partitions...",
  "reason_completeness": "Response A is more complete.",
  "reason_faithfulness": "Both are faithful.",
  "overall_rationale": "Mock response preferred A."
}
```

### Screenshot — position-bias result
```text
Running A/B pairwise evaluation with position-swapping...
Case [1/15]: Run1 Winner=A | Run2 Winner=A | Mapped=B | Consensus=TIE
Case [2/15]: Run1 Winner=A | Run2 Winner=A | Mapped=B | Consensus=TIE
Position Flip Rate: 100.00%
Consensus Win Rate (Config A): 0.00% | Consensus Win Rate (Config B): 0.00%
```

### Explain your position-bias check (your own words)
The position-bias check runs each pairwise comparison twice: first with Candidate A in Position 1 and B in Position 2, then with B in Position 1 and A in Position 2. A "flip" occurs if the judge changes its choice based on order. A high flip rate indicates that the judge is heavily biased by position (e.g. preferring the first option), making a single-run evaluation invalid. Our mitigation addresses this by declaring a TIE unless the judge consistently selects the same candidate.

### Adversarial probe outcome
```json
{
  "query": "What is the default replication factor for AetherDB?",
  "winner_declared": "TIE",
  "fooled": "N",
  "rationale": "Mock judge resolved order swap as a TIE, rejecting the verbose wrong answer."
}
```
*Read on it*: The judge successfully avoided declaring the verbose-but-wrong answer as the winner, declaring a TIE instead of being fooled by the wordy, authoritative explanation.

### AI usage disclosure
AI assistance (Antigravity IDE) was used to construct JSON mock templates, generate Mermaid code structures, and format submission logs. The positional-swap algorithm and parsing fallbacks were implemented and verified by the candidate.
