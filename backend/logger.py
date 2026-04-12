from backend.database import get_connection

def log_action(user, action):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO logs (user, action) VALUES (?, ?)",
        (user, action)
    )

    conn.commit()
    conn.close()
    