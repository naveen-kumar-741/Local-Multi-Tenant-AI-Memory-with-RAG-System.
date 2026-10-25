from fastapi import FastAPI, UploadFile, File
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

app = FastAPI()


# ============================================================
# Configuration
# ============================================================

MEMORY_COLLECTION = "memory"
KNOWLEDGE_COLLECTION = "knowledge"

VECTOR_SIZE = 384

# Used when retrieving information for answering questions
DEFAULT_SCORE_THRESHOLD = 0.3

# Used when finding potentially related memories
MEMORY_RELATIONSHIP_THRESHOLD = 0.5

# Used specifically for duplicate detection
MEMORY_DUPLICATE_THRESHOLD = 0.75

# Number of related memories inspected by the memory manager
MEMORY_RELATIONSHIP_LIMIT = 5

# Weight for vector similarity vs. memory importance
# final_score = (ALPHA * similarity_score) + ((1 - ALPHA) * importance)
IMPORTANCE_ALPHA = 0.7

# Knowledge Chunking Parameters (Phase 8.3)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "phi3:mini"


# ============================================================
# Qdrant
# ============================================================

qdrant = QdrantClient(":memory:")

qdrant.recreate_collection(
    collection_name=MEMORY_COLLECTION,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)

qdrant.recreate_collection(
    collection_name=KNOWLEDGE_COLLECTION,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)


# ============================================================
# Embedding Model
# ============================================================

model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# Request Models
# ============================================================


class MemoryRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    use_importance_weight: bool = True


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


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
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )


class SupersedeMemoryRequest(BaseModel):
    old_memory_id: str
    new_memory_id: str


class ProcessMemoryRequest(BaseModel):
    text: str
    memory_type: str = "fact"
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    source: str = "conversation"


# ============================================================
# Response Models
# ============================================================


class ConflictDetectionResult(BaseModel):
    conflict: bool
    reason: str = ""


class MemoryRelationshipResult(BaseModel):
    relationship: str
    reason: str = ""


class MemoryUpdateResult(BaseModel):
    memory_id: str
    text: str
    type: str
    importance: float
    created_at: str
    updated_at: str
    status: str
    supersedes: str | None = None


# ============================================================
# Memory Analyzer Model
# ============================================================


class MemoryCandidate(BaseModel):
    remember: bool
    type: str | None = None
    text: str | None = None
    importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# Utility Functions
# ============================================================


def create_embedding(text: str):
    """
    Convert text into an embedding vector.
    """
    return model.encode(text).tolist()


def current_timestamp():
    """
    Return current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def chunk_text_with_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks with character offsets.
    Phase 8.3 implementation.
    """
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

    chunks = []
    step = chunk_size - chunk_overlap
    text_length = len(text)

    start = 0
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_text,
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
# Memory Storage
# ============================================================


def store_memory(
    text: str,
    memory_type: str = "fact",
    importance: float = 0.5,
    source: str = "manual",
    supersedes: str | None = None,
):
    """
    Store a memory in the memory collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    timestamp = current_timestamp()

    payload = {
        "text": text,
        "type": memory_type,
        "importance": importance,
        "source": source,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "active",
        "supersedes": supersedes,
    }

    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )

    return point_id


# ============================================================
# Memory Update
# ============================================================


def update_memory(
    memory_id: str,
    text: str,
    memory_type: str,
    importance: float,
):
    """
    Update an existing active memory.
    """

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
        raise ValueError(f"Memory '{memory_id}' is not active and cannot be updated.")

    created_at = existing_payload.get(
        "created_at",
        current_timestamp(),
    )

    supersedes = existing_payload.get("supersedes")

    updated_at = current_timestamp()

    embedding = create_embedding(text)

    payload = {
        "text": text,
        "type": memory_type,
        "importance": importance,
        "source": existing_payload.get(
            "source",
            "conversation",
        ),
        "created_at": created_at,
        "updated_at": updated_at,
        "status": "active",
        "supersedes": supersedes,
    }

    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[
            PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )

    return {
        "memory_id": memory_id,
        "text": text,
        "type": memory_type,
        "importance": importance,
        "created_at": created_at,
        "updated_at": updated_at,
        "status": "active",
        "supersedes": supersedes,
    }


# ============================================================
# Supersede Memory
# ============================================================


def supersede_memory(
    old_memory_id: str,
    new_memory_id: str,
):
    """
    Mark an existing memory as superseded and connect the
    new memory to the old memory.
    """

    if old_memory_id == new_memory_id:
        raise ValueError("Old and new memory IDs must be different.")

    old_results = qdrant.retrieve(
        collection_name=MEMORY_COLLECTION,
        ids=[old_memory_id],
        with_payload=True,
        with_vectors=False,
    )

    if not old_results:
        raise ValueError(f"Old memory '{old_memory_id}' was not found.")

    new_results = qdrant.retrieve(
        collection_name=MEMORY_COLLECTION,
        ids=[new_memory_id],
        with_payload=True,
        with_vectors=False,
    )

    if not new_results:
        raise ValueError(f"New memory '{new_memory_id}' was not found.")

    old_point = old_results[0]
    new_point = new_results[0]

    old_payload = old_point.payload or {}
    new_payload = new_point.payload or {}

    if old_payload.get("status") != "active":
        raise ValueError(f"Old memory '{old_memory_id}' is not active.")

    if new_payload.get("status") != "active":
        raise ValueError(f"New memory '{new_memory_id}' is not active.")

    old_updated_at = current_timestamp()

    updated_old_payload = {
        **old_payload,
        "status": "superseded",
        "updated_at": old_updated_at,
    }

    qdrant.set_payload(
        collection_name=MEMORY_COLLECTION,
        payload=updated_old_payload,
        points=[old_memory_id],
    )

    new_updated_at = current_timestamp()

    updated_new_payload = {
        **new_payload,
        "status": "active",
        "supersedes": old_memory_id,
        "updated_at": new_updated_at,
    }

    qdrant.set_payload(
        collection_name=MEMORY_COLLECTION,
        payload=updated_new_payload,
        points=[new_memory_id],
    )

    return {
        "old_memory_id": old_memory_id,
        "new_memory_id": new_memory_id,
        "old_status": "superseded",
        "new_status": "active",
        "supersedes": old_memory_id,
    }


# ============================================================
# Knowledge Storage
# ============================================================


def store_knowledge(
    text: str,
    file_name: str,
    chunk_index: int,
    start_char: int | None = None,
    end_char: int | None = None,
    page: int | None = None,
):
    """
    Store a knowledge chunk in the knowledge collection.
    Updated for Phase 8.3 with offset tracking.
    """

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
        "created_at": current_timestamp(),
    }

    qdrant.upsert(
        collection_name=KNOWLEDGE_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )

    return point_id


# ============================================================
# Query Contextualization
# ============================================================


def contextualize_query(
    query: str,
    history: list[ChatMessage],
) -> str:
    """
    Reformulate follow-up user queries into a standalone search query
    using previous chat history context.
    """

    if not history:
        return query

    formatted_history = "\n".join(
        f"{msg.role.upper()}: {msg.content}" for msg in history
    )

    prompt = f"""
You are a query rewrite system.

Given the chat history and the latest user query, rewrite the user query
into a complete, standalone question/statement that can be used for vector similarity retrieval.

DO NOT answer the question.
Return ONLY the reformulated query text.

Chat History:
{formatted_history}

Latest Query:
{query}

Standalone Query:
"""

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        rewritten = response["message"]["content"].strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return query


@app.post("/contextualize_query")
def contextualize_query_endpoint(request: ContextualizeQueryRequest):

    standalone = contextualize_query(
        query=request.query,
        history=request.history,
    )

    return {
        "original_query": request.query,
        "contextualized_query": standalone,
        "history_turns": len(request.history),
    }


# ============================================================
# Memory Search
# ============================================================


def calculate_weighted_score(
    similarity_score: float,
    importance: float,
    alpha: float = IMPORTANCE_ALPHA,
) -> float:
    return (alpha * similarity_score) + ((1.0 - alpha) * importance)


def search_memory(
    query: str,
    limit: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_importance_weight: bool = True,
):

    query_embedding = create_embedding(query)

    candidate_limit = limit * 3 if use_importance_weight else limit

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=candidate_limit,
        score_threshold=score_threshold,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="status",
                    match=MatchValue(value="active"),
                )
            ]
        ),
    )

    points = results.points

    if not use_importance_weight or not points:
        return points[:limit]

    ranked_points = []
    for point in points:
        importance = float(point.payload.get("importance", 0.5))
        similarity = float(point.score)
        weighted_score = calculate_weighted_score(similarity, importance)
        point.score = weighted_score
        ranked_points.append(point)

    ranked_points.sort(key=lambda p: p.score, reverse=True)
    return ranked_points[:limit]


# ============================================================
# Knowledge Search
# ============================================================


def search_knowledge(
    query: str,
    limit: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):

    query_embedding = create_embedding(query)

    results = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=score_threshold,
    )

    return results.points


# ============================================================
# Find Related Memories
# ============================================================


def find_related_memories(
    text: str,
    limit: int = MEMORY_RELATIONSHIP_LIMIT,
    relationship_threshold: float = MEMORY_RELATIONSHIP_THRESHOLD,
):

    embedding = create_embedding(text)

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=embedding,
        limit=limit,
        score_threshold=relationship_threshold,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="status",
                    match=MatchValue(value="active"),
                )
            ]
        ),
    )

    return results.points


@app.post("/find_related_memories")
def find_related_memories_endpoint(request: SearchRequest):

    results = find_related_memories(
        text=request.query,
        limit=request.limit,
        relationship_threshold=request.score_threshold,
    )

    return {
        "collection": MEMORY_COLLECTION,
        "relationship_threshold": request.score_threshold,
        "results": [
            {
                "id": str(point.id),
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "importance": point.payload.get("importance"),
                "status": point.payload.get("status"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# Duplicate Memory Detection
# ============================================================


def find_duplicate_memory(
    text: str,
    duplicate_threshold: float = MEMORY_DUPLICATE_THRESHOLD,
):

    results = find_related_memories(
        text=text,
        limit=1,
        relationship_threshold=duplicate_threshold,
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
):

    duplicate_result = find_duplicate_memory(text)

    if duplicate_result["is_duplicate"]:

        return {
            "memory_stored": False,
            "memory_duplicate": True,
            "memory_id": None,
            "existing_memory_id": duplicate_result["existing_memory_id"],
            "existing_memory_text": duplicate_result["existing_text"],
            "duplicate_score": duplicate_result["score"],
        }

    memory_id = store_memory(
        text=text,
        memory_type=memory_type,
        importance=importance,
        source=source,
    )

    return {
        "memory_stored": True,
        "memory_duplicate": False,
        "memory_id": memory_id,
        "existing_memory_id": None,
        "existing_memory_text": None,
        "duplicate_score": None,
    }


# ============================================================
# Conflict Detection
# ============================================================


def detect_memory_conflict(
    new_memory: str,
    existing_memory: str,
) -> ConflictDetectionResult:

    prompt = f"""
You are a memory conflict detection system.

Your ONLY task is to determine whether the NEW MEMORY
contradicts the EXISTING MEMORY.

Return ONLY valid JSON.

EXISTING MEMORY
{existing_memory}

NEW MEMORY
{new_memory}
"""

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )

        parsed = json.loads(response["message"]["content"].strip())
        conflict = parsed.get("conflict", False)
        reason = parsed.get("reason", "")

        return ConflictDetectionResult(conflict=conflict, reason=str(reason))

    except Exception as exc:

        return ConflictDetectionResult(
            conflict=False,
            reason=f"Conflict detection failed: {type(exc).__name__}",
        )


@app.post("/detect_memory_conflict")
def detect_memory_conflict_endpoint(request: ConflictDetectionRequest):

    result = detect_memory_conflict(
        new_memory=request.new_memory,
        existing_memory=request.existing_memory,
    )

    return {
        "existing_memory": request.existing_memory,
        "new_memory": request.new_memory,
        "conflict": result.conflict,
        "reason": result.reason,
    }


# ============================================================
# Relationship Classification
# ============================================================


def classify_memory_relationship(
    new_memory: str,
    existing_memory: str,
) -> MemoryRelationshipResult:

    prompt = f"""
You are a memory relationship classification system.

Classify relationship into: duplicate, update, conflict, or unrelated.

EXISTING MEMORY: {existing_memory}
NEW MEMORY: {new_memory}

Return ONLY valid JSON.
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
        reason = str(parsed.get("reason", ""))

        if rel not in {"duplicate", "update", "conflict", "unrelated"}:
            rel = "unrelated"

        return MemoryRelationshipResult(relationship=rel, reason=reason)

    except Exception as exc:

        return MemoryRelationshipResult(
            relationship="unrelated",
            reason=f"Classification failed: {type(exc).__name__}",
        )


@app.post("/classify_memory_relationship")
def classify_memory_relationship_endpoint(request: MemoryRelationshipRequest):

    result = classify_memory_relationship(
        new_memory=request.new_memory,
        existing_memory=request.existing_memory,
    )

    return {
        "existing_memory": request.existing_memory,
        "new_memory": request.new_memory,
        "relationship": result.relationship,
        "reason": result.reason,
    }


# ============================================================
# Memory Manager
# ============================================================


def process_memory_candidate(
    text: str,
    memory_type: str,
    importance: float,
    source: str = "conversation",
):

    related_memories = find_related_memories(
        text=text,
        limit=MEMORY_RELATIONSHIP_LIMIT,
        relationship_threshold=MEMORY_RELATIONSHIP_THRESHOLD,
    )

    if not related_memories:

        memory_id = store_memory(
            text=text,
            memory_type=memory_type,
            importance=importance,
            source=source,
        )

        return {
            "action": "created",
            "relationship": "unrelated",
            "memory_id": memory_id,
            "existing_memory_id": None,
            "existing_memory_text": None,
            "relationship_score": None,
            "reason": "No related active memory was found.",
        }

    classified_candidates = []

    for point in related_memories:

        existing_text = point.payload.get("text", "")

        relationship_result = classify_memory_relationship(
            new_memory=text,
            existing_memory=existing_text,
        )

        classified_candidates.append(
            {
                "point": point,
                "relationship": relationship_result.relationship,
                "reason": relationship_result.reason,
                "score": float(point.score),
            }
        )

    relationship_priority = {
        "duplicate": 4,
        "conflict": 3,
        "update": 2,
        "unrelated": 1,
    }

    classified_candidates.sort(
        key=lambda candidate: (
            relationship_priority.get(
                candidate["relationship"],
                0,
            ),
            candidate["score"],
        ),
        reverse=True,
    )

    selected = classified_candidates[0]

    selected_point = selected["point"]
    selected_relationship = selected["relationship"]
    selected_reason = selected["reason"]
    selected_score = selected["score"]

    existing_memory_id = str(selected_point.id)
    existing_memory_text = selected_point.payload.get("text")

    if selected_relationship == "duplicate":

        return {
            "action": "ignored",
            "relationship": "duplicate",
            "memory_id": None,
            "existing_memory_id": existing_memory_id,
            "existing_memory_text": existing_memory_text,
            "relationship_score": selected_score,
            "reason": selected_reason,
        }

    if selected_relationship == "update":

        updated = update_memory(
            memory_id=existing_memory_id,
            text=text,
            memory_type=memory_type,
            importance=importance,
        )

        return {
            "action": "updated",
            "relationship": "update",
            "memory_id": updated["memory_id"],
            "existing_memory_id": existing_memory_id,
            "existing_memory_text": existing_memory_text,
            "relationship_score": selected_score,
            "reason": selected_reason,
            "created_at": updated["created_at"],
            "updated_at": updated["updated_at"],
        }

    if selected_relationship == "conflict":

        new_memory_id = store_memory(
            text=text,
            memory_type=memory_type,
            importance=importance,
            source=source,
            supersedes=existing_memory_id,
        )

        supersede_result = supersede_memory(
            old_memory_id=existing_memory_id,
            new_memory_id=new_memory_id,
        )

        return {
            "action": "superseded",
            "relationship": "conflict",
            "memory_id": new_memory_id,
            "existing_memory_id": existing_memory_id,
            "existing_memory_text": existing_memory_text,
            "relationship_score": selected_score,
            "reason": selected_reason,
            "old_status": supersede_result["old_status"],
            "new_status": supersede_result["new_status"],
            "supersedes": existing_memory_id,
        }

    memory_id = store_memory(
        text=text,
        memory_type=memory_type,
        importance=importance,
        source=source,
    )

    return {
        "action": "created",
        "relationship": "unrelated",
        "memory_id": memory_id,
        "existing_memory_id": None,
        "existing_memory_text": None,
        "relationship_score": None,
        "reason": (
            "Related candidates were found, but none represented "
            "an actionable relationship."
        ),
    }


@app.post("/process_memory_candidate")
def process_memory_candidate_endpoint(request: ProcessMemoryRequest):

    result = process_memory_candidate(
        text=request.text,
        memory_type=request.memory_type,
        importance=request.importance,
        source=request.source,
    )

    return result


@app.put("/update_memory")
def update_memory_endpoint(request: MemoryUpdateRequest):

    try:

        result = update_memory(
            memory_id=request.memory_id,
            text=request.text,
            memory_type=request.memory_type,
            importance=request.importance,
        )

        return {
            "status": "updated",
            **result,
        }

    except ValueError as exc:

        return {
            "status": "error",
            "error": str(exc),
        }


@app.put("/supersede_memory")
def supersede_memory_endpoint(request: SupersedeMemoryRequest):

    try:

        result = supersede_memory(
            old_memory_id=request.old_memory_id,
            new_memory_id=request.new_memory_id,
        )

        return {
            "status": "superseded",
            **result,
        }

    except ValueError as exc:

        return {
            "status": "error",
            "error": str(exc),
        }


@app.post("/store")
def add_memory(request: MemoryRequest):

    result = store_memory_if_not_duplicate(
        text=request.text,
        memory_type="memory",
        importance=0.5,
        source="manual",
    )

    return {
        "status": ("duplicate" if result["memory_duplicate"] else "stored"),
        "collection": MEMORY_COLLECTION,
        **result,
    }


# ============================================================
# Knowledge File Upload (Phase 8.3 Overlapping Chunking)
# ============================================================


@app.post("/add_memory_file_upload")
async def add_memory_file_upload(
    file: UploadFile = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):

    content = await file.read()

    file_name = file.filename or "unknown"

    text = ""

    if file_name.lower().endswith(".pdf"):

        with pdfplumber.open(io.BytesIO(content)) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    else:

        try:

            text = content.decode("utf-8")

        except UnicodeDecodeError:

            return {"error": "File must be a text or PDF file"}

    if not text.strip():

        return {"error": "File is empty"}

    # Phase 8.3 Overlapping chunking
    chunks = chunk_text_with_overlap(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    stored_ids = []

    for item in chunks:

        point_id = store_knowledge(
            text=item["text"],
            file_name=file_name,
            chunk_index=item["chunk_index"],
            start_char=item["start_char"],
            end_char=item["end_char"],
        )

        stored_ids.append(point_id)

    return {
        "status": "stored",
        "collection": KNOWLEDGE_COLLECTION,
        "file_name": file_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks_stored": len(chunks),
        "ids": stored_ids,
    }


# ============================================================
# Search Endpoints
# ============================================================


@app.post("/search_memory")
def search_memory_endpoint(request: SearchRequest):

    results = search_memory(
        query=request.query,
        limit=request.limit,
        score_threshold=request.score_threshold,
        use_importance_weight=request.use_importance_weight,
    )

    return {
        "collection": MEMORY_COLLECTION,
        "score_threshold": request.score_threshold,
        "results": [
            {
                "id": str(point.id),
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "importance": point.payload.get("importance"),
                "status": point.payload.get("status"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


@app.post("/search_knowledge")
def search_knowledge_endpoint(request: SearchRequest):

    results = search_knowledge(
        query=request.query,
        limit=request.limit,
        score_threshold=request.score_threshold,
    )

    return {
        "collection": KNOWLEDGE_COLLECTION,
        "score_threshold": request.score_threshold,
        "results": [
            {
                "id": str(point.id),
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "file_name": point.payload.get("file_name"),
                "page": point.payload.get("page"),
                "chunk_index": point.payload.get("chunk_index"),
                "start_char": point.payload.get("start_char"),
                "end_char": point.payload.get("end_char"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# Memory Analyzer
# ============================================================


def analyze_memory(user_message: str) -> MemoryCandidate:

    prompt = f"""
You are a memory extraction system.

Your ONLY job is to analyze the user's message and determine
whether it contains durable information about the user.

USER MESSAGE:
{user_message}

Return ONLY JSON.
"""

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )

        parsed = json.loads(response["message"]["content"].strip())
        candidate = MemoryCandidate(**parsed)

    except Exception:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    allowed_types = {"preference", "fact", "instruction"}

    if (
        not candidate.remember
        or candidate.type not in allowed_types
        or not candidate.text
        or not candidate.text.strip()
    ):
        return MemoryCandidate(remember=False, type=None, text=None, importance=0.0)

    importance = max(0.0, min(1.0, candidate.importance))

    return MemoryCandidate(
        remember=True,
        type=candidate.type,
        text=candidate.text.strip(),
        importance=importance,
    )


# ============================================================
# Chat Endpoint
# ============================================================


@app.post("/chat")
def chat(request: ChatRequest):

    search_query = contextualize_query(
        query=request.query,
        history=request.history,
    )

    memory_results = search_memory(
        query=search_query,
        limit=request.top_k,
        score_threshold=request.score_threshold,
        use_importance_weight=True,
    )

    knowledge_results = search_knowledge(
        query=search_query,
        limit=request.top_k,
        score_threshold=request.score_threshold,
    )

    memory_context_parts = []
    for point in memory_results:
        part = f"[Score: {point.score:.3f}]\n{point.payload.get('text', '')}"
        memory_context_parts.append(part)

    memory_context = (
        "\n---\n".join(memory_context_parts)
        if memory_context_parts
        else "No relevant memory found."
    )

    knowledge_context_parts = []
    for point in knowledge_results:
        part = f"[Score: {point.score:.3f}]\n"
        if point.payload.get("file_name"):
            part += f"File: {point.payload['file_name']}"
        if point.payload.get("page") is not None:
            part += f" | Page: {point.payload['page']}"
        if point.payload.get("chunk_index") is not None:
            part += f" | Chunk: {point.payload['chunk_index']}"
        part += f"\n{point.payload.get('text', '')}"
        knowledge_context_parts.append(part)

    knowledge_context = (
        "\n---\n".join(knowledge_context_parts)
        if knowledge_context_parts
        else "No relevant knowledge found."
    )

    context = (
        "\n--- User Memory ---\n"
        f"{memory_context}\n"
        "\n--- Knowledge ---\n"
        f"{knowledge_context}\n"
    )

    system_message = f"""
You are a helpful AI assistant.

Use the provided memory and knowledge when they are relevant.

--- User Memory ---
{memory_context}

--- Knowledge ---
{knowledge_context}

--- End Context ---
"""

    llm_messages = [{"role": "system", "content": system_message}]

    for msg in request.history:
        llm_messages.append({"role": msg.role, "content": msg.content})

    llm_messages.append({"role": "user", "content": request.query})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=llm_messages,
    )

    answer = response["message"]["content"]

    memory_candidate = analyze_memory(request.query)

    memory_management = {
        "action": "none",
        "relationship": None,
        "memory_id": None,
        "existing_memory_id": None,
        "existing_memory_text": None,
        "relationship_score": None,
        "reason": None,
    }

    if memory_candidate.remember:
        memory_management = process_memory_candidate(
            text=memory_candidate.text,
            memory_type=memory_candidate.type,
            importance=memory_candidate.importance,
            source="conversation",
        )

    return {
        "response": answer,
        "original_query": request.query,
        "search_query_used": search_query,
        "score_threshold": request.score_threshold,
        "memory_results": len(memory_results),
        "knowledge_results": len(knowledge_results),
        "context_used": context,
        "memory_analysis": {
            "remember": memory_candidate.remember,
            "type": memory_candidate.type,
            "text": memory_candidate.text,
            "importance": memory_candidate.importance,
        },
        "memory_management": memory_management,
    }
