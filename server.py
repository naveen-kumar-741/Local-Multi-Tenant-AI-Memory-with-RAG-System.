from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
import uuid
import ollama
import pdfplumber
import io
import json
from datetime import datetime, timezone
import json
from pydantic import BaseModel, Field
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

app = FastAPI()

# Enable CORS for frontend dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Configuration
# ============================================================

MEMORY_COLLECTION = "memory"
KNOWLEDGE_COLLECTION = "knowledge"

VECTOR_SIZE = 384

DEFAULT_SCORE_THRESHOLD = 0.5
MEMORY_RELATIONSHIP_THRESHOLD = 0.5
MEMORY_DUPLICATE_THRESHOLD = 0.75
MEMORY_RELATIONSHIP_LIMIT = 5

IMPORTANCE_ALPHA = 0.7

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"


# ============================================================
# Qdrant & Embedding Setup (Phase 10: Persistent Disk Storage)
# ============================================================

qdrant = QdrantClient(path="./qdrant_storage")

existing_collections = [col.name for col in qdrant.get_collections().collections]

if MEMORY_COLLECTION not in existing_collections:
    qdrant.create_collection(
        collection_name=MEMORY_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

if KNOWLEDGE_COLLECTION not in existing_collections:
    qdrant.create_collection(
        collection_name=KNOWLEDGE_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# Request Models
# ============================================================


class MemoryRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    session_id: str | None = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    use_importance_weight: bool = True
    user_id: str = "default_user"
    session_id: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    user_id: str = "default_user"
    session_id: str | None = None


class ContextualizeQueryRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)


class ConflictDetectionRequest(BaseModel):
    new_memory: str
    existing_memory: str


class MemoryRelationshipRequest(BaseModel):
    new_memory: str
    existing_memory: str


class MemoryUpdateRequest(BaseModel):
    memory_id: str
    text: str
    memory_type: str = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class SupersedeMemoryRequest(BaseModel):
    old_memory_id: str
    new_memory_id: str


class ProcessMemoryRequest(BaseModel):
    text: str
    memory_type: str = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "conversation"
    user_id: str = "default_user"
    session_id: str | None = None


# ============================================================
# Response & Analyzer Models
# ============================================================


class ConflictDetectionResult(BaseModel):
    conflict: bool
    reason: str = ""


class MemoryRelationshipResult(BaseModel):
    relationship: str
    reason: str = ""


class MemoryCandidate(BaseModel):
    remember: bool
    type: str | None = None
    text: str | None = None
    importance: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryRewriteOutput(BaseModel):
    is_standalone: bool = Field(
        description="True if the query is asking a standalone question (including general questions about user preferences/memories). False ONLY if it uses pronouns like 'it', 'that', 'he' that require context."
    )
    standalone_query: str = Field(
        description="The query to use. If is_standalone is True, return the EXACT latest query."
    )


# ============================================================
# Utility Functions
# ============================================================


def create_embedding(text: str):
    return model.encode(text).tolist()


def current_timestamp():
    return datetime.now(timezone.utc).isoformat()


def chunk_text_with_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    step = chunk_size - chunk_overlap
    text_length = len(text)
    start = 0
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
            }
        )
        chunk_index += 1
        if end == text_length:
            break
        start += step

    return chunks


# ============================================================
# Storage & Retrieval Operations
# ============================================================


def store_memory(
    text: str,
    memory_type: str = "fact",
    importance: float = 0.5,
    source: str = "manual",
    supersedes: str | None = None,
    user_id: str = "default_user",
    session_id: str | None = None,
):
    embedding = create_embedding(text)
    point_id = str(uuid.uuid4())
    timestamp = current_timestamp()

    payload = {
        "text": text,
        "type": memory_type,
        "importance": importance,
        "source": source,
        "user_id": user_id,
        "session_id": session_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "active",
        "supersedes": supersedes,
    }

    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
    )
    return point_id


def update_memory(
    memory_id: str,
    text: str,
    memory_type: str,
    importance: float,
):
    results = qdrant.retrieve(
        collection_name=MEMORY_COLLECTION,
        ids=[memory_id],
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        raise ValueError(f"Memory '{memory_id}' was not found.")

    existing_point = results[0]
    existing_payload = existing_point.payload or {}
    if existing_payload.get("status") != "active":
        raise ValueError(f"Memory '{memory_id}' is not active.")

    created_at = existing_payload.get("created_at", current_timestamp())
    supersedes = existing_payload.get("supersedes")
    user_id = existing_payload.get("user_id", "default_user")
    session_id = existing_payload.get("session_id")
    updated_at = current_timestamp()
    embedding = create_embedding(text)

    payload = {
        "text": text,
        "type": memory_type,
        "importance": importance,
        "source": existing_payload.get("source", "conversation"),
        "user_id": user_id,
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "status": "active",
        "supersedes": supersedes,
    }

    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[PointStruct(id=memory_id, vector=embedding, payload=payload)],
    )
    return {
        "memory_id": memory_id,
        "text": text,
        "type": memory_type,
        "importance": importance,
        "user_id": user_id,
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "status": "active",
        "supersedes": supersedes,
    }


def supersede_memory(old_memory_id: str, new_memory_id: str):
    if old_memory_id == new_memory_id:
        raise ValueError("Old and new memory IDs must be different.")

    old_results = qdrant.retrieve(
        collection_name=MEMORY_COLLECTION,
        ids=[old_memory_id],
        with_payload=True,
        with_vectors=False,
    )
    new_results = qdrant.retrieve(
        collection_name=MEMORY_COLLECTION,
        ids=[new_memory_id],
        with_payload=True,
        with_vectors=False,
    )

    if not old_results or not new_results:
        raise ValueError("Old or new memory was not found.")

    old_point, new_point = old_results[0], new_results[0]
    timestamp = current_timestamp()

    qdrant.set_payload(
        collection_name=MEMORY_COLLECTION,
        payload={**old_point.payload, "status": "superseded", "updated_at": timestamp},
        points=[old_memory_id],
    )
    qdrant.set_payload(
        collection_name=MEMORY_COLLECTION,
        payload={
            **new_point.payload,
            "status": "active",
            "supersedes": old_memory_id,
            "updated_at": timestamp,
        },
        points=[new_memory_id],
    )

    return {
        "old_memory_id": old_memory_id,
        "new_memory_id": new_memory_id,
        "old_status": "superseded",
        "new_status": "active",
        "supersedes": old_memory_id,
    }


def store_knowledge(
    text: str,
    file_name: str,
    chunk_index: int,
    start_char: int | None = None,
    end_char: int | None = None,
    page: int | None = None,
    user_id: str = "default_user",
):
    embedding = create_embedding(text)
    point_id = str(uuid.uuid4())
    payload = {
        "text": text,
        "type": "knowledge",
        "file_name": file_name,
        "page": page,
        "chunk_index": chunk_index,
        "start_char": start_char,
        "end_char": end_char,
        "user_id": user_id,
        "created_at": current_timestamp(),
    }
    qdrant.upsert(
        collection_name=KNOWLEDGE_COLLECTION,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
    )
    return point_id


# ============================================================
# Search & Filtering Helpers
# ============================================================


def contextualize_query(
    query: str,
    history: list[ChatMessage],
) -> str:
    """
    Production contextualization using JSON classification.
    Stops the LLM from replacing summary queries with previous user turns.
    """
    if not history:
        return query

    # ONLY extract past USER messages. Ignore long Assistant responses.
    user_turns = [msg.content for msg in history if msg.role == "user"][-3:]

    if not user_turns:
        return query

    formatted_user_history = "\n".join(
        f"- Turn {i+1}: {text}" for i, text in enumerate(user_turns[:-1])
    )

    prompt = f"""
You are a vector search query classifier.

TASK:
Analyze the LATEST USER QUERY.
Determine if it is already a complete question or a general request (e.g. asking what they prefer, asking about their memories, or asking a broad question).

RULES:
1. If the LATEST USER QUERY is asking a question or asking to recall memories/preferences, set `is_standalone` to true. Do NOT replace it with past sentences.
2. Set `is_standalone` to false ONLY if the latest query contains ambiguous pronouns ("how do I do it?", "tell me more about that").

Past User Messages:
{formatted_user_history}

Latest User Query:
{query}

Return JSON matching this schema:
{{"is_standalone": true|false, "standalone_query": "string"}}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )

        parsed = json.loads(response["message"]["content"].strip())

        # If the LLM classifies it as standalone, return original query unchanged
        if parsed.get("is_standalone", False):
            return query

        rewritten = parsed.get("standalone_query", "").strip().strip("\"'")
        if rewritten:
            return rewritten

    except Exception:
        pass

    return query


def rerank_knowledge_results(points: list, query: str, limit: int) -> list:
    if not points:
        return []

    scored_points = []
    seen_snippets = set()

    for point in points:
        text = point.payload.get("text", "").strip()
        snippet = text[:50].lower()
        if snippet in seen_snippets:
            continue
        seen_snippets.add(snippet)

        density_bonus = min(0.05, len(text) / 10000.0)
        point.score = float(point.score) + density_bonus
        scored_points.append(point)

    scored_points.sort(key=lambda p: p.score, reverse=True)
    return scored_points[:limit]


def calculate_weighted_score(
    similarity_score: float, importance: float, alpha: float = IMPORTANCE_ALPHA
) -> float:
    return (alpha * similarity_score) + ((1.0 - alpha) * importance)


def search_memory(
    query: str,
    limit: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_importance_weight: bool = True,
    user_id: str = "default_user",
):
    query_embedding = create_embedding(query)
    candidate_limit = limit * 3 if use_importance_weight else limit

    filter_conditions = [
        FieldCondition(key="status", match=MatchValue(value="active")),
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ]

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=candidate_limit,
        score_threshold=score_threshold,
        query_filter=Filter(must=filter_conditions),
    )

    points = results.points
    if not use_importance_weight or not points:
        return points[:limit]

    ranked_points = []
    for point in points:
        importance = float(point.payload.get("importance", 0.5))
        similarity = float(point.score)
        point.score = calculate_weighted_score(similarity, importance)
        ranked_points.append(point)

    ranked_points.sort(key=lambda p: p.score, reverse=True)
    return ranked_points[:limit]


def search_knowledge(
    query: str,
    limit: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    user_id: str = "default_user",
):
    query_embedding = create_embedding(query)

    filter_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ]

    results = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=query_embedding,
        limit=limit * 2,
        score_threshold=score_threshold,
        query_filter=Filter(must=filter_conditions),
    )
    return rerank_knowledge_results(results.points, query, limit)


def find_related_memories(
    text: str,
    limit: int = MEMORY_RELATIONSHIP_LIMIT,
    relationship_threshold: float = MEMORY_RELATIONSHIP_THRESHOLD,
    user_id: str = "default_user",
):
    embedding = create_embedding(text)
    filter_conditions = [
        FieldCondition(key="status", match=MatchValue(value="active")),
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ]
    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=embedding,
        limit=limit,
        score_threshold=relationship_threshold,
        query_filter=Filter(must=filter_conditions),
    )
    return results.points


def find_duplicate_memory(
    text: str,
    user_id: str = "default_user",
    duplicate_threshold: float = MEMORY_DUPLICATE_THRESHOLD,
):
    results = find_related_memories(
        text, limit=1, relationship_threshold=duplicate_threshold, user_id=user_id
    )
    if not results:
        return {
            "is_duplicate": False,
            "existing_memory_id": None,
            "existing_text": None,
            "score": None,
        }
    point = results[0]
    return {
        "is_duplicate": True,
        "existing_memory_id": str(point.id),
        "existing_text": point.payload.get("text"),
        "score": float(point.score),
    }


def store_memory_if_not_duplicate(
    text: str,
    memory_type: str,
    importance: float,
    source: str,
    user_id: str = "default_user",
    session_id: str | None = None,
):
    dup = find_duplicate_memory(text, user_id=user_id)
    if dup["is_duplicate"]:
        return {"memory_stored": False, "memory_duplicate": True, **dup}
    mem_id = store_memory(
        text, memory_type, importance, source, user_id=user_id, session_id=session_id
    )
    return {"memory_stored": True, "memory_duplicate": False, "memory_id": mem_id}


# ============================================================
# Conflict & Relationship Detection
# ============================================================


def detect_memory_conflict(
    new_memory: str, existing_memory: str
) -> ConflictDetectionResult:
    prompt = f"""
Determine whether NEW MEMORY contradicts EXISTING MEMORY. Return ONLY JSON: {{"conflict": true}} or {{"conflict": false}}.
EXISTING MEMORY: {existing_memory}
NEW MEMORY: {new_memory}
"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        parsed = json.loads(response["message"]["content"].strip())
        return ConflictDetectionResult(
            conflict=parsed.get("conflict", False), reason=str(parsed.get("reason", ""))
        )
    except Exception as exc:
        return ConflictDetectionResult(conflict=False, reason=str(exc))


def classify_memory_relationship(
    new_memory: str, existing_memory: str
) -> MemoryRelationshipResult:
    prompt = f"""
Classify relationship into: duplicate, update, conflict, or unrelated. Return ONLY JSON.
EXISTING: {existing_memory}
NEW: {new_memory}
"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        parsed = json.loads(response["message"]["content"].strip())
        rel = parsed.get("relationship", "unrelated")
        if rel not in {"duplicate", "update", "conflict", "unrelated"}:
            rel = "unrelated"
        return MemoryRelationshipResult(
            relationship=rel, reason=str(parsed.get("reason", ""))
        )
    except Exception as exc:
        return MemoryRelationshipResult(relationship="unrelated", reason=str(exc))


def process_memory_candidate(
    text: str,
    memory_type: str,
    importance: float,
    source: str = "conversation",
    user_id: str = "default_user",
    session_id: str | None = None,
):
    related = find_related_memories(
        text,
        limit=MEMORY_RELATIONSHIP_LIMIT,
        relationship_threshold=MEMORY_RELATIONSHIP_THRESHOLD,
        user_id=user_id,
    )
    if not related:
        mem_id = store_memory(
            text,
            memory_type,
            importance,
            source,
            user_id=user_id,
            session_id=session_id,
        )
        return {"action": "created", "relationship": "unrelated", "memory_id": mem_id}

    classified = []
    for point in related:
        rel_res = classify_memory_relationship(text, point.payload.get("text", ""))
        classified.append(
            {
                "point": point,
                "relationship": rel_res.relationship,
                "score": float(point.score),
            }
        )

    priority = {"duplicate": 4, "conflict": 3, "update": 2, "unrelated": 1}
    classified.sort(
        key=lambda c: (priority.get(c["relationship"], 0), c["score"]), reverse=True
    )
    selected = classified[0]

    sel_point, sel_rel = selected["point"], selected["relationship"]
    ex_id, ex_text = str(sel_point.id), sel_point.payload.get("text")

    if sel_rel == "duplicate":
        return {
            "action": "ignored",
            "relationship": "duplicate",
            "existing_memory_id": ex_id,
        }
    if sel_rel == "update":
        updated = update_memory(ex_id, text, memory_type, importance)
        return {"action": "updated", "relationship": "update", **updated}
    if sel_rel == "conflict":
        new_id = store_memory(
            text,
            memory_type,
            importance,
            source,
            supersedes=ex_id,
            user_id=user_id,
            session_id=session_id,
        )
        supersede_memory(ex_id, new_id)
        return {
            "action": "superseded",
            "relationship": "conflict",
            "memory_id": new_id,
            "existing_memory_id": ex_id,
        }

    mem_id = store_memory(
        text, memory_type, importance, source, user_id=user_id, session_id=session_id
    )
    return {"action": "created", "relationship": "unrelated", "memory_id": mem_id}


# ============================================================
# API Endpoints
# ============================================================


@app.post("/contextualize_query")
def contextualize_query_endpoint(request: ContextualizeQueryRequest):
    return {
        "original_query": request.query,
        "contextualized_query": contextualize_query(request.query, request.history),
    }


@app.post("/detect_memory_conflict")
def detect_memory_conflict_endpoint(request: ConflictDetectionRequest):
    res = detect_memory_conflict(request.new_memory, request.existing_memory)
    return {"conflict": res.conflict, "reason": res.reason}


@app.post("/classify_memory_relationship")
def classify_memory_relationship_endpoint(request: MemoryRelationshipRequest):
    res = classify_memory_relationship(request.new_memory, request.existing_memory)
    return {"relationship": res.relationship, "reason": res.reason}


@app.post("/process_memory_candidate")
def process_memory_candidate_endpoint(request: ProcessMemoryRequest):
    return process_memory_candidate(
        request.text,
        request.memory_type,
        request.importance,
        request.source,
        user_id=request.user_id,
        session_id=request.session_id,
    )


@app.put("/update_memory")
def update_memory_endpoint(request: MemoryUpdateRequest):
    try:
        return {
            "status": "updated",
            **update_memory(
                request.memory_id, request.text, request.memory_type, request.importance
            ),
        }
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.put("/supersede_memory")
def supersede_memory_endpoint(request: SupersedeMemoryRequest):
    try:
        return {
            "status": "superseded",
            **supersede_memory(request.old_memory_id, request.new_memory_id),
        }
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@app.post("/store")
def add_memory(request: MemoryRequest):
    return store_memory_if_not_duplicate(
        request.text,
        "memory",
        0.5,
        "manual",
        user_id=request.user_id,
        session_id=request.session_id,
    )


@app.post("/add_memory_file_upload")
async def add_memory_file_upload(
    file: UploadFile = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    user_id: str = "default_user",
):
    content = await file.read()
    file_name = file.filename or "unknown"
    text = ""

    if file_name.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                pt = page.extract_text()
                if pt:
                    text += pt + "\n"
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "File must be text or PDF"}

    if not text.strip():
        return {"error": "File is empty"}

    chunks = chunk_text_with_overlap(text, chunk_size, chunk_overlap)
    stored_ids = [
        store_knowledge(
            c["text"],
            file_name,
            c["chunk_index"],
            c["start_char"],
            c["end_char"],
            user_id=user_id,
        )
        for c in chunks
    ]

    return {
        "status": "stored",
        "user_id": user_id,
        "chunks_stored": len(chunks),
        "ids": stored_ids,
    }


@app.post("/search_memory")
def search_memory_endpoint(request: SearchRequest):
    results = search_memory(
        request.query,
        request.limit,
        request.score_threshold,
        request.use_importance_weight,
        user_id=request.user_id,
    )
    return {
        "user_id": request.user_id,
        "results": [
            {"id": str(p.id), "text": p.payload.get("text"), "score": float(p.score)}
            for p in results
        ],
    }


@app.post("/search_knowledge")
def search_knowledge_endpoint(request: SearchRequest):
    results = search_knowledge(
        request.query, request.limit, request.score_threshold, user_id=request.user_id
    )
    return {
        "user_id": request.user_id,
        "results": [
            {"id": str(p.id), "text": p.payload.get("text"), "score": float(p.score)}
            for p in results
        ],
    }


@app.post("/chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    search_query = contextualize_query(request.query, request.history)
    memory_results = search_memory(
        search_query,
        request.top_k,
        request.score_threshold,
        True,
        user_id=request.user_id,
    )
    knowledge_results = search_knowledge(
        search_query, request.top_k, request.score_threshold, user_id=request.user_id
    )

    memory_context = (
        "\n---\n".join(
            [
                f"[Score: {p.score:.3f}]\n{p.payload.get('text', '')[:300]}"
                for p in memory_results
            ]
        )
        or "No memory found."
    )
    knowledge_context = (
        "\n---\n".join(
            [
                f"[Score: {p.score:.3f}]\n{p.payload.get('text', '')[:300]}"
                for p in knowledge_results
            ]
        )
        or "No knowledge found."
    )

    system_message = f"""You are a concise assistant. Keep answers brief (1-2 sentences).
--- Memory ---
{memory_context}
--- Knowledge ---
{knowledge_context}"""

    llm_messages = [{"role": "system", "content": system_message}]
    for msg in request.history:
        llm_messages.append({"role": msg.role, "content": msg.content})
    llm_messages.append({"role": "user", "content": request.query})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=llm_messages,
        options={
            "num_predict": 80,  # Keep generated responses concise
            "num_ctx": 1024,  # Minimal VRAM allocation
            "temperature": 0.2,
        },
    )
    answer = response["message"]["content"]

    # RUN MEMORY EXTRACTION IN THE BACKGROUND (Frees up response time)
    def background_memory_extraction(query, uid, sid):
        cand = analyze_memory(query)
        if cand.remember:
            process_memory_candidate(
                cand.text,
                cand.type,
                cand.importance,
                "conversation",
                user_id=uid,
                session_id=sid,
            )

    background_tasks.add_task(
        background_memory_extraction, request.query, request.user_id, request.session_id
    )

    return {
        "response": answer,
        "user_id": request.user_id,
        "search_query_used": search_query,
        "memory_results_count": len(memory_results),
        "knowledge_results_count": len(knowledge_results),
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    search_query = contextualize_query(request.query, request.history)
    memory_results = search_memory(
        search_query,
        request.top_k,
        request.score_threshold,
        True,
        user_id=request.user_id,
    )
    knowledge_results = search_knowledge(
        search_query, request.top_k, request.score_threshold, user_id=request.user_id
    )

    memory_context = (
        "\n---\n".join(
            [
                f"[Score: {p.score:.3f}]\n{p.payload.get('text', '')[:300]}"
                for p in memory_results
            ]
        )
        or "No memory found."
    )
    knowledge_context = (
        "\n---\n".join(
            [
                f"[Score: {p.score:.3f}]\n{p.payload.get('text', '')[:300]}"
                for p in knowledge_results
            ]
        )
        or "No knowledge found."
    )

    system_message = f"""You are a concise assistant. Keep answers brief (1-2 sentences).
--- Memory ---
{memory_context}
--- Knowledge ---
{knowledge_context}"""

    llm_messages = [{"role": "system", "content": system_message}]
    for msg in request.history:
        llm_messages.append({"role": msg.role, "content": msg.content})
    llm_messages.append({"role": "user", "content": request.query})

    # Schedule background memory extraction without blocking streaming
    def background_memory_extraction(query, uid, sid):
        cand = analyze_memory(query)
        if cand.remember:
            process_memory_candidate(
                cand.text,
                cand.type,
                cand.importance,
                "conversation",
                user_id=uid,
                session_id=sid,
            )

    background_tasks.add_task(
        background_memory_extraction, request.query, request.user_id, request.session_id
    )

    def event_stream():
        response_stream = ollama.chat(
            model=LLM_MODEL,
            messages=llm_messages,
            stream=True,  # Enable token streaming
            options={"num_predict": 80, "num_ctx": 1024, "temperature": 0.2},
        )
        for chunk in response_stream:
            yield chunk["message"]["content"]

    return StreamingResponse(event_stream(), media_type="text/plain")


def analyze_memory(user_message: str) -> MemoryCandidate:
    prompt = f"""
You are a memory extraction system.
Analyze the user's message and determine if it contains durable, long-term facts, preferences, personal details, or habits (e.g. technology preferences, favorite foods, job role, locations).

MESSAGE: "{user_message}"

Rules:
1. If the message states a preference, fact, goal, or personal detail about the user, set "remember" to true.
2. Extract the core concise statement for "text" (e.g. "Prefers Android over iPhone for mobile").
3. Assign "type" as "preference", "fact", or "habit".
4. Assign "importance" between 0.5 and 0.9.

Return JSON strictly matching this schema:
{{
    "remember": true | false,
    "type": "preference" | "fact" | "habit" | null,
    "text": "extracted memory sentence or null",
    "importance": float
}}
"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        parsed = json.loads(response["message"]["content"].strip())
        return MemoryCandidate(**parsed)
    except Exception:
        return MemoryCandidate(remember=False)
