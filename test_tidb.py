import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

host = os.getenv("TIDB_HOST")
port = os.getenv("TIDB_PORT", "4000")
user = os.getenv("TIDB_USER")
pwd  = os.getenv("TIDB_PASSWORD")
db   = os.getenv("TIDB_DB", "support_ai")
ca   = os.getenv("TIDB_SSL_CA")  # path to tidb-ca.pem

# IMPORTANT: pass SSL CA to PyMySQL via connect_args
engine = create_engine(
    f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}",
    connect_args={"ssl": {"ca": ca}},
    pool_pre_ping=True,            # keeps long-lived connections healthy
    pool_recycle=300               # avoids stale TCP connections
)

try:
    with engine.begin() as conn:
        current = conn.execute(text("SELECT DATABASE()")).scalar_one()
        print("Connected to DB:", current)

        rows = conn.execute(text("""
            SELECT id, external_id, sentiment
            FROM conversations
            ORDER BY id DESC
            LIMIT 3
        """)).fetchall()
        print("Recent rows:", rows)

        conn.execute(text("""
            INSERT INTO conversations
                (external_id, customer_id, source, raw_text, sentiment, embedding, meta)
            VALUES
                (:external_id, :customer_id, 'chatbot', :raw_text, :sentiment, NULL, JSON_OBJECT('channel','web'))
        """), {
            "external_id": "conv_pytest",
            "customer_id": "cust_py",
            "raw_text": "Test insert from Python. All good.",
            "sentiment": 0.1000
        })

        print("Insert OK.")
except Exception as e:
    # Print the exact underlying DB error to know what's wrong
    print("Connection or query failed ->", repr(e))
    raise
