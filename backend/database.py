import sqlite3

DB_PATH = "backend/app.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# ---------------- INIT DB ----------------
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # 👤 USERS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        login TEXT UNIQUE,
        password TEXT,
        role TEXT,
        class TEXT
    )
    """)

    # 📜 LOGS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        action TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()