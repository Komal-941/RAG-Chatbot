# database.py
import sqlite3
import streamlit as st


# Use st.cache_resource to ensure we only connect to the DB once
@st.cache_resource
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn


def init_db():
    """Initializes the database and creates the messages table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print("Database initialized.")


def save_message(chat_id: str, role: str, content: str):
    """Saves a message to the database for a specific chat session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content)
    )
    conn.commit()


def load_messages(chat_id: str) -> list:
    """Loads all messages for a specific chat session from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
    messages = [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]
    return messages


def clear_chat(chat_id: str):
    """Deletes all messages for a specific chat session from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.commit()


def get_chat_sessions() -> list:
    """Retrieves a list of unique chat_id's from the database, most recent first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT chat_id FROM messages ORDER BY timestamp DESC")
    sessions = [row["chat_id"] for row in cursor.fetchall()]
    return sessions


# --- Call this once when the app starts ---
init_db()
