from backend.database import get_connection
from backend.logger import log_action


# 🔐 LOGIN
def authenticate_user(login, password):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT login, role, class FROM users WHERE login=? AND password=?", (login, password))
    user = cur.fetchone()

    conn.close()

    if user:
        log_action(login, "login")
        return {
            "login": user[0],
            "role": user[1],
            "class": user[2]
        }

    return None


# ➕ CREATE
def create_user(login, password, role, class_name):

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (login, password, role, class) VALUES (?, ?, ?, ?)",
            (login, password, role, class_name)
        )
        conn.commit()

        log_action(login, "user_created")

        return True
    except:
        return False
    finally:
        conn.close()


# 📋 READ
def get_all_users():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT login, role, class FROM users")
    users = cur.fetchall()

    conn.close()

    return [{"login": u[0], "role": u[1], "class": u[2]} for u in users]


# ❌ DELETE
def delete_user(login):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM users WHERE login=?", (login,))
    conn.commit()
    conn.close()

    log_action(login, "user_deleted")


# ✏️ UPDATE
def update_user(login, role, class_name):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE users SET role=?, class=? WHERE login=?",
        (role, class_name, login)
    )

    conn.commit()
    conn.close()

    log_action(login, "user_updated")


# 📜 LOGS
def get_logs():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT user, action, timestamp FROM logs ORDER BY timestamp DESC")
    logs = cur.fetchall()

    conn.close()

    return logs