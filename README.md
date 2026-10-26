# Local Multi-Tenant AI Memory + RAG System

A hardened, low-latency multi-tenant local RAG (Retrieval-Augmented Generation) and asynchronous user memory management system built on **FastAPI**, **Qdrant**, and **Ollama**.

Designed for zero-cloud reliance, high throughput, and strict hardware alignment, this architecture provides real-time SSE streaming for chat responses while running long-term memory extraction asynchronously in background pipelines.

---

## ✨ Features

- **Sub-2s Latency Pipeline:** Token-by-token SSE streaming masks LLM inference time and drops Time-To-First-Token (TTFT) to **< 300ms**.
- **100% GPU Execution:** Hardware-aligned context limits (`num_ctx`, `num_predict`) prevent VRAM-to-CPU spillover on standard consumer GPUs.
- **Dual Vector Store Architecture (Qdrant):**
  - `memory`: Isolated user preferences, facts, and short/long-term context.
  - `knowledge`: Document chunks with reranking for domain knowledge base lookup.
- **Asynchronous Memory Engine:** `/chat` streams immediately while FastAPI `BackgroundTasks` analyze and consolidate memory in the background without blocking the response.
- **Interactive Testbench & Inspector:** Built-in HTML & React dashboards for real-time state inspection, document uploads, query reformulation testing, and memory conflict classification.

---

## ⚡ Prerequisites

Before getting started, ensure you have the following installed on your host machine:

- **Python 3.10+**: Runtime environment for FastAPI.
- **Ollama**: Local LLM server ([Download Ollama](https://ollama.com)).
- **Docker & Docker Compose** (Optional): For running Qdrant in a containerized environment.
- **NVIDIA GPU / Apple Silicon**: Dedicated VRAM is recommended to maintain sub-2s generation speeds.

---

## 🚀 Quickstart

### Step 1: Clone the Repository & Setup Python Environment

```bash
git clone https://github.com/your-username/ai-memory-rag-system.git
cd ai-memory-rag-system

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Qdrant Vector Store

Run Qdrant using Docker:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Step 3: Pull & Configure the GPU-Optimized LLM

Pull `qwen2.5:1.5b` via Ollama for guaranteed 100% GPU offloading:

```bash
ollama pull qwen2.5:1.5b
```

Verify that Ollama offloads 100% to GPU during inference:

```bash
ollama ps
# Output must show: 100% GPU / 100% CUDA (0% CPU)
```

### Step 4: Launch FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

_The API is now running at `http://localhost:8000`._

### Step 5: Test via Web Dashboard

Open `ai_memory_rag_system_dashboard.html` in any web browser to access the interactive chat, inspector pane, memory manager, and document upload tool.

---

## 🛠 Tech Stack

- **Backend Framework:** FastAPI (Python)
- **LLM Engine:** Ollama (`qwen2.5:1.5b` with strict `num_ctx` and `num_predict` caps)
- **Vector Database:** Qdrant (disk-persistent dual collections: `memory` & `knowledge`)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Frontend:** Vanilla HTML/JS Dashboard & React Component

---

## 📡 API Endpoints Summary

| Method | Endpoint                  | Description                                                                  |
| :----- | :------------------------ | :--------------------------------------------------------------------------- |
| `POST` | `/chat/stream`            | Real-time SSE streaming chat with RAG context & background memory extraction |
| `POST` | `/store`                  | Manually store a structured user memory                                      |
| `POST` | `/search_memory`          | Query user memories with importance weighting                                |
| `POST` | `/search_knowledge`       | Search ingested document chunks                                              |
| `POST` | `/add_memory_file_upload` | Chunk and index PDF/TXT files into knowledge base                            |
| `POST` | `/detect_memory_conflict` | Test and classify memory conflicts/relationships                             |
| `POST` | `/contextualize_query`    | Standalone query rewriter test endpoint                                      |

---

## 📄 License

MIT License
