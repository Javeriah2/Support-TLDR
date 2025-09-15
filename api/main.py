import os, json, requests
from typing import Optional

from sqlalchemy import text as sql_text
from fastapi import Query, HTTPException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
import csv, io, json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi import Body




# Load .env once at startup
load_dotenv()

# --- Read env vars ---
host = os.getenv("TIDB_HOST")
port = os.getenv("TIDB_PORT", "4000")
user = os.getenv("TIDB_USER")
pwd  = os.getenv("TIDB_PASSWORD")
db   = os.getenv("TIDB_DB", "support_ai")
ca   = os.getenv("TIDB_SSL_CA", "tidb-ca.pem")  # path to CA file
EMBED_PROVIDER   = os.getenv("EMBED_PROVIDER", "ollama")
OLLAMA_BASE      = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM        = int(os.getenv("EMBED_DIM", "768"))
EMBED_MAX_CHARS  = int(os.getenv("EMBED_MAX_CHARS", "2000"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "3"))
ESCALATE_SENTIMENT_THRESHOLD = float(os.getenv("ESCALATE_SENTIMENT_THRESHOLD", "-0.5"))
ESCALATE_MIN_TEXT_LEN = int(os.getenv("ESCALATE_MIN_TEXT_LEN", "0"))
AUTO_EMAIL_ENABLED = os.getenv("AUTO_EMAIL_ENABLED", "false").lower() == "true"


from fastapi import Body
SUMMARIZER_PROVIDER = os.getenv("SUMMARIZER_PROVIDER", "ollama")
OLLAMA_SUMMARY_MODEL = os.getenv("OLLAMA_SUMMARY_MODEL", "llama3.2:3b-instruct")
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "220"))


# --- Create a single engine (connection pool) ---
# Why a pool? Reuses connections; faster & reliable for multiple requests.
engine = create_engine(
    f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}",
    connect_args={"ssl": {"ca": ca}},  # TLS for TiDB Cloud
    pool_pre_ping=True,                # auto-heal dead connections
    pool_recycle=300                   # avoid stale TCP
)

analyzer = SentimentIntensityAnalyzer()


app = FastAPI(title="Smart Escalation API", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev: allow all; we’ll lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pydantic models -----
class ConversationIn(BaseModel):
    external_id: Optional[str] = None
    customer_id: Optional[str] = None
    source: str = Field(default="chatbot", description="e.g., 'chatbot'")
    raw_text: str
    sentiment: Optional[float] = Field(default=None, description="-1.0 (very negative) to +1.0 (very positive)")

def free_sentiment_score(text: str | None) -> float | None:
    if not text or not text.strip():
        return None
    # VADER compound in [-1, 1]
    return float(analyzer.polarity_scores(text)["compound"])

def embed_with_ollama(text: str) -> list[float]:
    if not text:
        return []
    if len(text) > EMBED_MAX_CHARS:
        text = text[:EMBED_MAX_CHARS]

    url = f"{OLLAMA_BASE}/api/embeddings"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text, "prompt": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "embedding" in data:
        vec = data["embedding"]
    elif "embeddings" in data and data["embeddings"]:
        vec = data["embeddings"][0]
    else:
        raise RuntimeError(f"Unexpected Ollama response: keys={list(data.keys())}")

    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"Embedding dim {len(vec)} != expected {EMBED_DIM}")
    return vec

def summarize_with_ollama(text: str, similar: list[str]) -> str:
    if not text or not text.strip():
        return "No content to summarize."
    similar_block = "\n".join([f"- {s[:200]}" for s in similar if s])  # keep short
    prompt = f"""You are a support escalation assistant.
Summarize the customer's situation briefly and propose next steps.

Customer text:
{text}

Similar past cases (snippets):
{similar_block}

Return:
1) TL;DR (2-3 lines, no fluff)
2) Pain points (bullets)
3) Recommended next steps (bullets)
Keep it under 120 words, neutral, actionable."""
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": OLLAMA_SUMMARY_MODEL,
            "messages": [
                {"role": "system", "content": "Be concise, clear, and helpful."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": SUMMARY_MAX_TOKENS},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    # Ollama returns either message.content or response depending on version
    return (data.get("message", {}) or {}).get("content") or data.get("response", "")



# ----- Endpoints -----

@app.get("/health")
def health():
    """
    Returns 'ok' and confirms DB connectivity by asking the DB for its name.
    Why: proves env vars, SSL, networking, and credentials are correct.
    If we didn't do a DB check here, the endpoint could say OK even if DB is broken.
    """
    try:
        with engine.begin() as conn:
            dbname = conn.execute(sql_text("SELECT DATABASE()")).scalar_one()
        return {"status": "ok", "db": dbname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {repr(e)}")

@app.get("/conversations/recent")
def recent_conversations():
    """
    Returns the last 5 conversations (basic fields).
    Why: quick read to show the API can fetch from TiDB and return JSON.
    """
    try:
        with engine.begin() as conn:
            rows = conn.execute(sql_text("""
                SELECT id, external_id, customer_id, source, raw_text, sentiment
                FROM conversations
                ORDER BY id DESC
                LIMIT 5
            """)).mappings().all()
        # .mappings() gives dict-like rows → easy to jsonify
        return {"items": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {repr(e)}")

@app.post("/conversations")
def create_conversation(conv: ConversationIn):
    """
    Inserts a new conversation (embedding=NULL for now).
    Why: proves we can safely write to TiDB from the API using parameterized SQL.
    If we didn't parameterize, we'd risk SQL injection and bugs around quoting.
    """
    params = conv.model_dump()
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("""
                INSERT INTO conversations
                    (external_id, customer_id, source, raw_text, sentiment, embedding, meta)
                VALUES
                    (:external_id, :customer_id, :source, :raw_text, :sentiment, NULL, JSON_OBJECT('channel','web'))
            """), params)
            new_id = conn.execute(sql_text("SELECT LAST_INSERT_ID()")).scalar_one()
        return {"id": new_id, "ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {repr(e)}")

@app.post("/upload/json")
def upload_json(payload: list[dict] | dict = Body(..., embed=False)):
    """
    Accepts JSON:
      - Either a list of items: [{external_id, customer_id, raw_text, ...}, ...]
      - Or an object: {"items": [ ...same as above... ]}
    Each item: raw_text is required; other fields optional.
    """
    # Normalize payload to a list
    items = payload if isinstance(payload, list) else payload.get("items", [])
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="Invalid JSON format. Expect list or {items: list}.")

    rows = []
    for it in items:
        # basic shape safeguards
        raw_text = (it.get("raw_text") or "").strip()
        if not raw_text:
            # skip empty rows quietly
            continue
        sentiment = it.get("sentiment")
        if sentiment is None:
            sentiment = free_sentiment_score(raw_text)

        rows.append({
            "external_id": it.get("external_id"),
            "customer_id": it.get("customer_id"),
            "source": it.get("source", "chatbot"),
            "raw_text": raw_text,
            "sentiment": sentiment
        })

    if not rows:
        return {"ok": True, "inserted": 0}

    try:
        with engine.begin() as conn:
            conn.execute(sql_text("""
                INSERT INTO conversations
                    (external_id, customer_id, source, raw_text, sentiment, embedding, meta)
                VALUES
                    (:external_id, :customer_id, :source, :raw_text, :sentiment, NULL, JSON_OBJECT('ingest','json'))
            """), rows)  # SQLAlchemy can bulk-insert a list of dicts
        return {"ok": True, "inserted": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {repr(e)}")

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with a single 'file'.
    CSV headers expected: external_id, customer_id, source, raw_text, sentiment
    JSON can be a list or {items: [...]} with same fields.
    """
    content = await file.read()
    name = (file.filename or "").lower()

    # Try to detect by extension; fallback to content sniffing
    def parse_json_bytes(b: bytes):
        data = json.loads(b.decode("utf-8"))
        return data if isinstance(data, list) else data.get("items", [])

    def parse_csv_bytes(b: bytes):
        text = b.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    try:
        if name.endswith(".json"):
            items = parse_json_bytes(content)
        elif name.endswith(".csv"):
            items = parse_csv_bytes(content)
        else:
            # try json first, then csv
            try:
                items = parse_json_bytes(content)
            except Exception:
                items = parse_csv_bytes(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to parse file. Use JSON or CSV.")

    # Normalize + score sentiment
    rows = []
    for it in items:
        raw_text = (it.get("raw_text") or "").strip()
        if not raw_text:
            continue
        sentiment = it.get("sentiment")
        if isinstance(sentiment, str) and sentiment.strip() != "":
            try:
                sentiment = float(sentiment)
            except:
                sentiment = None
        if sentiment is None:
            sentiment = free_sentiment_score(raw_text)

        rows.append({
            "external_id": it.get("external_id"),
            "customer_id": it.get("customer_id"),
            "source": it.get("source", "chatbot"),
            "raw_text": raw_text,
            "sentiment": sentiment
        })

    if not rows:
        return {"ok": True, "inserted": 0}

    try:
        with engine.begin() as conn:
            conn.execute(sql_text("""
                INSERT INTO conversations
                    (external_id, customer_id, source, raw_text, sentiment, embedding, meta)
                VALUES
                    (:external_id, :customer_id, :source, :raw_text, :sentiment, NULL, JSON_OBJECT('ingest','file'))
            """), rows)
        return {"ok": True, "inserted": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {repr(e)}")

@app.post("/embeddings/backfill")
def backfill_embeddings(limit: int = 50):
    if EMBED_PROVIDER != "ollama":
        raise HTTPException(status_code=400, detail="Set EMBED_PROVIDER=ollama in .env")

    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT id, raw_text
            FROM conversations
            WHERE embedding IS NULL AND raw_text IS NOT NULL AND raw_text != ''
            ORDER BY id DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()

    if not rows:
        return {"ok": True, "updated": 0}

    updated = 0
    with engine.begin() as conn:
        for r in rows:
            try:
                vec = embed_with_ollama(r["raw_text"])
                conn.execute(sql_text("""
                    UPDATE conversations SET embedding = :embedding WHERE id = :id
                """), {"embedding": json.dumps(vec), "id": r["id"]})
                updated += 1
            except Exception as e:
                # skip bad row; continue
                pass

    return {"ok": True, "updated": updated, "scanned": len(rows)}

@app.get("/search/similar")
def search_similar(
    query_text: str = Query(..., min_length=1, alias="text"),
    k: int = Query(default=SIMILARITY_TOP_K, ge=1, le=20),
):
    if EMBED_PROVIDER != "ollama":
        raise HTTPException(status_code=400, detail="Set EMBED_PROVIDER=ollama in .env")

    vec = embed_with_ollama(query_text)
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT id, external_id, customer_id, source, raw_text, sentiment,
                   VEC_COSINE_DISTANCE(embedding, :qvec) AS score
            FROM conversations
            WHERE embedding IS NOT NULL
            ORDER BY score ASC
            LIMIT :k
        """), {"qvec": json.dumps(vec), "k": k}).mappings().all()

    return {"items": [dict(r) for r in rows], "query": query_text}

@app.post("/context/summary")
def context_summary(body: dict = Body(...)):
    if SUMMARIZER_PROVIDER != "ollama":
        raise HTTPException(status_code=400, detail="Set SUMMARIZER_PROVIDER=ollama in .env")

    text_in = (body.get("text") or "").strip()
    k = int(body.get("k", SIMILARITY_TOP_K))
    if not text_in:
        raise HTTPException(status_code=400, detail="text is required")

    # find top-k similar for context (re-use your embed + TiDB search)
    qvec = json.dumps(embed_with_ollama(text_in))
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT id, raw_text, sentiment,
                   VEC_COSINE_DISTANCE(embedding, :q) AS score
            FROM conversations
            WHERE embedding IS NOT NULL
            ORDER BY score ASC
            LIMIT :k
        """), {"q": qvec, "k": k}).mappings().all()

    similar_snippets = [r["raw_text"] for r in rows]
    summary = summarize_with_ollama(text_in, similar_snippets)
    return {"summary": summary, "similar": [dict(r) for r in rows]}

@app.post("/escalate/check")
def escalate_check(id: int, force: bool = False):
    # 1) load conversation
    with engine.begin() as conn:
        row = conn.execute(sql_text("""
            SELECT id, external_id, customer_id, raw_text, sentiment, escalated_at
            FROM conversations WHERE id=:id
        """), {"id": id}).mappings().first()
    if not row:
        raise HTTPException(404, "conversation not found")

    # 2) simple checks (explainable)
    raw = (row["raw_text"] or "").strip()
    sent = None if row["sentiment"] is None else float(row["sentiment"])

    if not force:
        if raw == "" or len(raw) < ESCALATE_MIN_TEXT_LEN:
            return {"escalate": False, "reason": "too_short_or_empty"}
        if sent is None:
            return {"escalate": False, "reason": "no_sentiment"}
        if sent > ESCALATE_SENTIMENT_THRESHOLD:
            return {"escalate": False, "reason": "sentiment_above_threshold"}

    # 3) dedupe: avoid re-escalating unless forced
    if row["escalated_at"] is not None and not force:
        return {"escalate": False, "reason": "already_escalated"}

    # 4) context: top-3 similar
    qvec = json.dumps(embed_with_ollama(raw))
    with engine.begin() as conn:
        sims = conn.execute(sql_text("""
            SELECT id, raw_text, sentiment,
                   VEC_COSINE_DISTANCE(embedding, :q) AS score
            FROM conversations
            WHERE embedding IS NOT NULL
            ORDER BY score ASC
            LIMIT 3
        """), {"q": qvec}).mappings().all()

    # 5) generate a concise, actionable summary
    summary = summarize_with_ollama(raw, [s["raw_text"] for s in sims])

    # 6) persist on conversation + create escalation record
    with engine.begin() as conn:
        conn.execute(sql_text("""
          UPDATE conversations
          SET escalated_at = NOW(), handoff_summary = :sum
          WHERE id = :id
        """), {"sum": summary, "id": id})
        conn.execute(sql_text("""
          INSERT INTO escalations (conversation_id, sentiment, summary, status)
          VALUES (:id, :sent, :sum, 'open')
        """), {"id": id, "sent": sent, "sum": summary})

    # (Optional) if you implemented auto-email in 3.2, you could call it here.

    return {
        "escalate": True,
        "reason": "threshold_met" if not force else "forced",
        "summary": summary,
        "similar": [dict(s) for s in sims],
        "saved": {"conversation": True, "escalation": True}
    }

@app.get("/escalations/recent")
def escalations_recent(limit: int = 20):
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
          SELECT e.id, e.conversation_id, e.created_at, e.sentiment, e.status,
                 c.external_id, c.customer_id
          FROM escalations e
          JOIN conversations c ON c.id = e.conversation_id
          ORDER BY e.id DESC
          LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"items": [dict(r) for r in rows]}
