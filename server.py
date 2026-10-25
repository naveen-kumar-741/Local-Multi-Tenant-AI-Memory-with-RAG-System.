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

# Phase 8.1 Configuration: Weight for vector similarity vs. memory importance
# final_score = (ALPHA * similarity_score) + ((1 - ALPHA) * importance)
IMPORTANCE_ALPHA = 0.7

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


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


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

    The existing memory ID is preserved.

    created_at is preserved.
    updated_at is changed.
    status remains active.
    supersedes is preserved.

    The vector is regenerated because the text has changed.
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

    # --------------------------------------------------------
    # Mark old memory as superseded
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Link new memory to old memory
    # --------------------------------------------------------

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
    page: int | None = None,
):
    """
    Store a knowledge chunk in the knowledge collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    payload = {
        "text": text,
        "type": "knowledge",
        "file_name": file_name,
        "page": page,
        "chunk_index": chunk_index,
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
# Memory Search
# ============================================================


def calculate_weighted_score(
    similarity_score: float,
    importance: float,
    alpha: float = IMPORTANCE_ALPHA,
) -> float:
    """
    Calculate importance-weighted score combining similarity and stored importance.
    Phase 8.1 enhancement.
    """
    return (alpha * similarity_score) + ((1.0 - alpha) * importance)


def search_memory(
    query: str,
    limit: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_importance_weight: bool = True,
):
    """
    Search only active memories from the memory collection.

    Superseded memories remain in Qdrant for history/debugging
    but are excluded from normal retrieval.

    Phase 8.1: Applies importance weighting to re-rank search results.
    """

    query_embedding = create_embedding(query)

    # Retrieve candidate set with broader limit if weighted ranking is enabled
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

    # Re-rank candidates using importance-weighted score
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
    """
    Search only the knowledge collection.
    """

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
    """
    Find active memories that are semantically related to the
    supplied memory.

    This function ONLY finds candidate memories.

    It does NOT determine whether the relationship is:
        - duplicate
        - update
        - conflict
        - unrelated
    """

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


# ============================================================
# Related Memory Endpoint
# ============================================================


@app.post("/find_related_memories")
def find_related_memories_endpoint(request: SearchRequest):
    """
    Find active memories that are semantically related to the
    supplied query.
    """

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
    """
    Detect whether the supplied memory is semantically similar
    enough to an existing active memory to be considered a
    duplicate.

    Phase 7.3 behavior.
    """

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


# ============================================================
# Store Memory If Not Duplicate
# ============================================================


def store_memory_if_not_duplicate(
    text: str,
    memory_type: str,
    importance: float,
    source: str,
):
    """
    Check for an existing similar memory before storing.

    This function is retained for backward compatibility with
    the existing manual memory endpoint.

    Automatic conversation memory now uses process_memory_candidate().
    """

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
    """
    Determine whether a new memory contradicts an existing memory.

    This function ONLY detects conflict.
    """

    prompt = f"""
You are a memory conflict detection system.

Your ONLY task is to determine whether the NEW MEMORY
contradicts the EXISTING MEMORY.

Return ONLY valid JSON.

The JSON must contain:

{{
    "conflict": true
}}

or:

{{
    "conflict": false
}}

A "reason" field is optional.

============================================================
WHAT IS A CONFLICT?
============================================================

A conflict exists when the new memory says that a previous
user preference, choice, tool, technology, behavior, or state
has been replaced, reversed, or is no longer true.

Treat these as conflicts:

- switched from X to Y
- moved from X to Y
- changed from X to Y
- no longer use X
- stopped using X
- instead of X, I use Y
- I now prefer Y when the existing memory says the user
  prefers X
- replacement of one tool, language, framework, editor,
  preference, or choice with another

============================================================
IMPORTANT
============================================================

Do NOT require words such as:

- switched
- replaced
- instead
- no longer

to appear explicitly.

For example:

Existing:
I prefer JavaScript for my projects.

New:
I prefer TypeScript for my projects.

This IS a conflict.

============================================================
WHEN IT IS NOT A CONFLICT
============================================================

Return conflict=false when:

1. The memories are duplicates or paraphrases.

2. The new memory adds information.

3. Both memories can remain true simultaneously.

4. The memories are unrelated.

Different technologies do NOT automatically conflict.

For example:

Existing:
I mainly work with React.

New:
I prefer TypeScript for my projects.

These can both be true.

============================================================
EXAMPLES
============================================================

Example 1:

Existing:
I prefer JavaScript for my projects.

New:
I prefer TypeScript for my projects.

Output:
{{
    "conflict": true
}}

------------------------------------------------------------

Example 2:

Existing:
I prefer JavaScript for my projects.

New:
I've switched to TypeScript for all my projects.

Output:
{{
    "conflict": true
}}

------------------------------------------------------------

Example 3:

Existing:
I prefer TypeScript for my projects.

New:
I prefer using TypeScript in my projects.

Output:
{{
    "conflict": false
}}

------------------------------------------------------------

Example 4:

Existing:
I mainly work with React.

New:
I prefer TypeScript for my projects.

Output:
{{
    "conflict": false
}}

------------------------------------------------------------

Example 5:

Existing:
I mainly work with React.

New:
I mainly work with React and TypeScript.

Output:
{{
    "conflict": false
}}

------------------------------------------------------------

Example 6:

Existing:
I use VS Code for development.

New:
I've switched to Cursor for development.

Output:
{{
    "conflict": true
}}

============================================================
EXISTING MEMORY
============================================================

{existing_memory}

============================================================
NEW MEMORY
============================================================

{new_memory}

============================================================

Return ONLY valid JSON.
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
            format="json",
            options={
                "temperature": 0,
            },
        )

        raw_content = response["message"]["content"].strip()

        parsed = json.loads(raw_content)

        if "conflict" not in parsed:

            return ConflictDetectionResult(
                conflict=False,
                reason=("LLM response did not contain a " "'conflict' field."),
            )

        conflict_value = parsed["conflict"]

        if isinstance(conflict_value, bool):

            conflict = conflict_value

        elif isinstance(conflict_value, str):

            normalized = conflict_value.strip().lower()

            if normalized == "true":

                conflict = True

            elif normalized == "false":

                conflict = False

            else:

                return ConflictDetectionResult(
                    conflict=False,
                    reason=("LLM returned an invalid value for " "'conflict'."),
                )

        else:

            return ConflictDetectionResult(
                conflict=False,
                reason=("LLM returned an invalid type for " "'conflict'."),
            )

        reason = parsed.get("reason", "")

        if not isinstance(reason, str):

            reason = str(reason)

        return ConflictDetectionResult(
            conflict=conflict,
            reason=reason.strip(),
        )

    except json.JSONDecodeError:

        return ConflictDetectionResult(
            conflict=False,
            reason="LLM returned invalid JSON.",
        )

    except Exception as exc:

        return ConflictDetectionResult(
            conflict=False,
            reason=(f"Conflict detection failed: " f"{type(exc).__name__}"),
        )


# ============================================================
# Conflict Detection Endpoint
# ============================================================


@app.post("/detect_memory_conflict")
def detect_memory_conflict_endpoint(
    request: ConflictDetectionRequest,
):
    """
    Test whether two memories contradict each other.

    This endpoint does NOT modify Qdrant.
    """

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
    """
    Classify the relationship between a new memory and an
    existing memory.

    Possible relationships:

        - duplicate
        - update
        - conflict
        - unrelated

    This function ONLY classifies the relationship.

    It does NOT modify Qdrant.
    """

    prompt = f"""
You are a memory relationship classification system.

Your ONLY task is to classify the relationship between:

1. EXISTING MEMORY
2. NEW MEMORY

Return ONLY valid JSON.

The relationship MUST be exactly one of:

- duplicate
- update
- conflict
- unrelated

JSON format:

{{
    "relationship": "duplicate",
    "reason": "short explanation"
}}

============================================================
RELATIONSHIP DEFINITIONS
============================================================

DUPLICATE
---------

The new memory expresses essentially the same information
as the existing memory.

Different wording does NOT make it an update.

Example:

Existing:
I prefer TypeScript for my projects.

New:
I prefer using TypeScript in my projects.

Relationship:
duplicate

------------------------------------------------------------

UPDATE
------

The new memory adds, refines, or expands information from
the existing memory without contradicting it.

Example:

Existing:
I mainly work with React.

New:
I mainly work with React and TypeScript.

Relationship:
update

Another example:

Existing:
I work with React.

New:
I mainly work with React for frontend projects.

Relationship:
update

------------------------------------------------------------

CONFLICT
--------

The new memory contradicts, replaces, reverses, or invalidates
the existing memory.

A conflict requires a genuine contradiction or replacement.

DO NOT classify memories as conflict merely because they
mention different technologies.

Different technologies can be compatible.

For example:

Existing:
I mainly work with React.

New:
I prefer TypeScript for my projects.

These can both be true.

Therefore:

Relationship:
unrelated

------------------------------------------------------------

UNRELATED
---------

The two memories describe separate information and do not
represent the same fact, preference, instruction, or state.

They may still be semantically related at a broad topic level.

Example:

Existing:
I prefer TypeScript for my projects.

New:
I enjoy playing soccer.

Relationship:
unrelated

Another important example:

Existing:
I mainly work with React.

New:
I prefer TypeScript for my projects.

Relationship:
unrelated

Reason:
React and TypeScript are different technologies and the
statements can both be true.

============================================================
IMPORTANT CLASSIFICATION RULES
============================================================

Rule 1:
duplicate is more specific than update.

If the new memory contains essentially the same information,
use duplicate.

Rule 2:
update means the new memory adds meaningful information to
the existing memory.

Rule 3:
conflict means the new memory replaces, reverses, or
contradicts the existing information.

Rule 4:
Different technologies, tools, frameworks, or concepts do
NOT automatically conflict.

Only classify them as conflict when the statements establish
an actual replacement or incompatible choice.

For example:

Existing:
I mainly work with React.

New:
I prefer TypeScript.

This is NOT conflict.

------------------------------------------------------------

Existing:
I use VS Code for development.

New:
I've switched to Cursor for development.

This IS conflict because Cursor replaces VS Code.

------------------------------------------------------------

Rule 5:
If both memories can remain true simultaneously, do NOT use
conflict.

Rule 6:
If the memories describe clearly different information, use
unrelated even if their general topics are similar.

============================================================
EXAMPLES
============================================================

Example 1:

Existing:
I prefer TypeScript for my projects.

New:
I prefer using TypeScript in my projects.

Output:
{{
    "relationship": "duplicate",
    "reason": "Both memories express the same preference."
}}

------------------------------------------------------------

Example 2:

Existing:
I mainly work with React.

New:
I mainly work with React and TypeScript.

Output:
{{
    "relationship": "update",
    "reason": "The new memory adds TypeScript to the existing development information."
}}

------------------------------------------------------------

Example 3:

Existing:
I prefer JavaScript for my projects.

New:
I prefer TypeScript for my projects.

Output:
{{
    "relationship": "conflict",
    "reason": "The new programming-language preference replaces JavaScript with TypeScript."
}}

------------------------------------------------------------

Example 4:

Existing:
I prefer TypeScript for my projects.

New:
I enjoy playing soccer.

Output:
{{
    "relationship": "unrelated",
    "reason": "The memories describe unrelated information."
}}

------------------------------------------------------------

Example 5:

Existing:
I use VS Code for development.

New:
I've switched to Cursor for development.

Output:
{{
    "relationship": "conflict",
    "reason": "The new editor choice replaces VS Code."
}}

------------------------------------------------------------

Example 6:

Existing:
I mainly work with React.

New:
I prefer TypeScript for my projects.

Output:
{{
    "relationship": "unrelated",
    "reason": "React and TypeScript are different technologies and both statements can be true."
}}

------------------------------------------------------------

Example 7:

Existing:
I work with React.

New:
I work with React for frontend projects.

Output:
{{
    "relationship": "update",
    "reason": "The new memory adds context to the existing React information."
}}

============================================================
EXISTING MEMORY
============================================================

{existing_memory}

============================================================
NEW MEMORY
============================================================

{new_memory}

============================================================

Return ONLY valid JSON.
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
            format="json",
            options={
                "temperature": 0,
            },
        )

        raw_content = response["message"]["content"].strip()

        parsed = json.loads(raw_content)

        relationship = parsed.get("relationship")

        allowed_relationships = {
            "duplicate",
            "update",
            "conflict",
            "unrelated",
        }

        if relationship not in allowed_relationships:

            return MemoryRelationshipResult(
                relationship="unrelated",
                reason=("LLM returned an invalid relationship."),
            )

        reason = parsed.get("reason", "")

        if not isinstance(reason, str):

            reason = str(reason)

        return MemoryRelationshipResult(
            relationship=relationship,
            reason=reason.strip(),
        )

    except json.JSONDecodeError:

        return MemoryRelationshipResult(
            relationship="unrelated",
            reason="LLM returned invalid JSON.",
        )

    except Exception as exc:

        return MemoryRelationshipResult(
            relationship="unrelated",
            reason=(f"Relationship classification failed: " f"{type(exc).__name__}"),
        )


# ============================================================
# Relationship Classification Endpoint
# ============================================================


@app.post("/classify_memory_relationship")
def classify_memory_relationship_endpoint(
    request: MemoryRelationshipRequest,
):
    """
    Classify the relationship between two memories.

    This endpoint does NOT modify Qdrant.
    """

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
    """
    Process a new memory candidate through the complete memory
    lifecycle.

    Flow:

        1. Find related active memories.
        2. Classify each candidate.
        3. Choose the strongest actionable relationship.
        4. Apply the corresponding lifecycle operation.

    Possible actions:

        duplicate
            Ignore the new memory.

        update
            Update the existing memory using the same memory ID.

        conflict
            Create a new memory and supersede the old memory.

        unrelated
            Create a new independent memory.

    This is the central memory-management function used by /chat.
    """

    related_memories = find_related_memories(
        text=text,
        limit=MEMORY_RELATIONSHIP_LIMIT,
        relationship_threshold=MEMORY_RELATIONSHIP_THRESHOLD,
    )

    # --------------------------------------------------------
    # No related candidates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Classify every candidate
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Relationship priority
    # --------------------------------------------------------
    #
    # When multiple related memories are returned:
    #
    # duplicate > conflict > update > unrelated
    #
    # Within the same relationship type, use the highest
    # semantic similarity score.
    # --------------------------------------------------------

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

    # ========================================================
    # Duplicate
    # ========================================================

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

    # ========================================================
    # Update
    # ========================================================

    if selected_relationship == "update":

        existing_payload = selected_point.payload or {}

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

    # ========================================================
    # Conflict
    # ========================================================

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

    # ========================================================
    # Unrelated
    # ========================================================

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


# ============================================================
# Memory Manager Endpoint
# ============================================================


@app.post("/process_memory_candidate")
def process_memory_candidate_endpoint(
    request: ProcessMemoryRequest,
):
    """
    Test the complete memory-management lifecycle.

    This endpoint can be used to test Phase 7.8 without
    calling /chat.
    """

    result = process_memory_candidate(
        text=request.text,
        memory_type=request.memory_type,
        importance=request.importance,
        source=request.source,
    )

    return result


# ============================================================
# Memory Update Endpoint
# ============================================================


@app.put("/update_memory")
def update_memory_endpoint(
    request: MemoryUpdateRequest,
):
    """
    Update an existing active memory.

    This endpoint is for testing Phase 7.6.
    """

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


# ============================================================
# Supersede Memory Endpoint
# ============================================================


@app.put("/supersede_memory")
def supersede_memory_endpoint(
    request: SupersedeMemoryRequest,
):
    """
    Mark an old active memory as superseded and link the new
    active memory to it.
    """

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


# ============================================================
# Manual Memory Endpoint
# ============================================================


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
# Knowledge File Upload
# ============================================================


@app.post("/add_memory_file_upload")
async def add_memory_file_upload(
    file: UploadFile = File(...),
):

    content = await file.read()

    file_name = file.filename or "unknown"

    text = ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if file_name.lower().endswith(".pdf"):

        with pdfplumber.open(io.BytesIO(content)) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    else:

        try:

            text = content.decode("utf-8")

        except UnicodeDecodeError:

            return {"error": "File must be a text or PDF file"}

    # --------------------------------------------------------
    # Empty File
    # --------------------------------------------------------

    if not text.strip():

        return {"error": "File is empty"}

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]

    stored_ids = []

    for chunk_index, chunk in enumerate(chunks):

        point_id = store_knowledge(
            text=chunk,
            file_name=file_name,
            chunk_index=chunk_index,
        )

        stored_ids.append(point_id)

    return {
        "status": "stored",
        "collection": KNOWLEDGE_COLLECTION,
        "file_name": file_name,
        "chunks_stored": len(chunks),
        "ids": stored_ids,
    }


# ============================================================
# Search Memory Endpoint
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


# ============================================================
# Search Knowledge Endpoint
# ============================================================


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
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# Memory Analyzer
# ============================================================


def analyze_memory(user_message: str) -> MemoryCandidate:
    """
    Analyze a user message and determine whether it contains
    durable information worth remembering.
    """

    prompt = f"""
You are a memory extraction system.

Your ONLY job is to analyze the user's message and determine
whether it contains durable information about the user that
would be useful in future conversations.

DO NOT answer the user's question.

DO NOT provide explanations.

Return ONLY JSON.

------------------------------------------------------------
WHEN TO REMEMBER
------------------------------------------------------------

Remember the message if it contains:

1. A user preference.

Examples:

"I prefer TypeScript."
"I like concise explanations."
"I prefer dark mode."

2. A stable fact about the user.

Examples:

"I mainly work with React."
"I use VS Code for development."
"I work mostly on frontend projects."

3. A persistent instruction.

Examples:

"From now on, always explain programming concepts with examples."
"Always use TypeScript in your examples."
"Keep your answers concise."
"Don't use overly complicated explanations."

4. A change to an existing preference or workflow.

Examples:

"I've switched to TypeScript."
"I no longer use JavaScript."
"I now use VS Code."

------------------------------------------------------------
WHEN NOT TO REMEMBER
------------------------------------------------------------

Do NOT remember:

- General knowledge questions
- One-time questions
- Greetings
- Thanks
- Temporary requests
- Requests to explain a topic
- Questions about unrelated facts
- Information that does not describe a durable user preference,
  fact, or instruction

Examples:

"What is React?"

"What is the capital of France?"

"How does Python work?"

"Thanks!"

"Can you explain recursion?"

------------------------------------------------------------
IMPORTANT NORMALIZATION RULE
------------------------------------------------------------

Convert the user's wording into a concise, reusable statement.

For technology preferences, prefer wording such as:

"User prefers TypeScript for projects."

instead of:

"User has switched exclusively to using TypeScript for
their projects."

For development facts:

"User mainly works with React."

For editor preferences:

"User uses VS Code for development."

For instructions:

"User prefers programming concepts to be explained with
simple examples."

------------------------------------------------------------
MEMORY TYPES
------------------------------------------------------------

Allowed types:

- preference
- fact
- instruction

------------------------------------------------------------
IMPORTANCE
------------------------------------------------------------

Use a value between 0 and 1.

Examples:

0.9 = very important persistent preference/instruction

0.7 = useful long-term fact

0.5 = relatively minor preference

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

If the message should be remembered:

{{
    "remember": true,
    "type": "preference",
    "text": "User prefers TypeScript for projects.",
    "importance": 0.8
}}

If the message should be ignored:

{{
    "remember": false,
    "type": null,
    "text": null,
    "importance": 0
}}

------------------------------------------------------------

USER MESSAGE:

{user_message}
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
            format="json",
        )

        raw_content = response["message"]["content"].strip()

        parsed = json.loads(raw_content)

        candidate = MemoryCandidate(**parsed)

    except Exception:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    allowed_types = {
        "preference",
        "fact",
        "instruction",
    }

    if not candidate.remember:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    if candidate.type not in allowed_types:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    if not candidate.text or not candidate.text.strip():

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    importance = max(
        0.0,
        min(1.0, candidate.importance),
    )

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

    # --------------------------------------------------------
    # 1. Retrieve Memory (with Importance Weighting)
    # --------------------------------------------------------

    memory_results = search_memory(
        query=request.query,
        limit=request.top_k,
        score_threshold=request.score_threshold,
        use_importance_weight=True,
    )

    # --------------------------------------------------------
    # 2. Retrieve Knowledge
    # --------------------------------------------------------

    knowledge_results = search_knowledge(
        query=request.query,
        limit=request.top_k,
        score_threshold=request.score_threshold,
    )

    # --------------------------------------------------------
    # 3. Build Memory Context
    # --------------------------------------------------------

    memory_context_parts = []

    for point in memory_results:

        part = f"[Score: {point.score:.3f}]\n" f"{point.payload.get('text', '')}"

        memory_context_parts.append(part)

    if memory_context_parts:

        memory_context = "\n---\n".join(memory_context_parts)

    else:

        memory_context = "No relevant memory found."

    # --------------------------------------------------------
    # 4. Build Knowledge Context
    # --------------------------------------------------------

    knowledge_context_parts = []

    for point in knowledge_results:

        part = f"[Score: {point.score:.3f}]\n"

        if point.payload.get("file_name"):

            part += f"File: " f"{point.payload['file_name']}"

        if point.payload.get("page") is not None:

            part += f" | Page: " f"{point.payload['page']}"

        if point.payload.get("chunk_index") is not None:

            part += f" | Chunk: " f"{point.payload['chunk_index']}"

        part += f"\n" f"{point.payload.get('text', '')}"

        knowledge_context_parts.append(part)

    if knowledge_context_parts:

        knowledge_context = "\n---\n".join(knowledge_context_parts)

    else:

        knowledge_context = "No relevant knowledge found."

    # --------------------------------------------------------
    # 5. Combined Context
    # --------------------------------------------------------

    context = (
        "\n--- User Memory ---\n"
        f"{memory_context}\n"
        "\n--- Knowledge ---\n"
        f"{knowledge_context}\n"
    )

    # --------------------------------------------------------
    # 6. Main LLM Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the provided memory and knowledge when they
are relevant to the user's question.

--- User Memory ---
{memory_context}

--- Knowledge ---
{knowledge_context}

--- End Context ---

Question:
{request.query}

Instructions:

- Use relevant memory when answering.
- Use relevant knowledge when answering.
- Do not mention the retrieval system.
- Do not invent information.
- If the context does not contain the answer,
  use your general knowledge when appropriate.
- Keep the answer concise and natural.
"""

    # --------------------------------------------------------
    # 7. Generate Answer
    # --------------------------------------------------------

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response["message"]["content"]

    # --------------------------------------------------------
    # 8. Analyze User Message
    # --------------------------------------------------------

    memory_candidate = analyze_memory(request.query)

    # --------------------------------------------------------
    # 9. Automatic Memory Management
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 10. Return Response
    # --------------------------------------------------------

    return {
        "response": answer,
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
