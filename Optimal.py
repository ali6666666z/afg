import streamlit as st
import sqlite3
from datetime import datetime

# Initialize database
def init_db():
    conn = sqlite3.connect("chatgpt_clone.db")
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Create chat history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

# Register user
def register_user(username, password):
    try:
        conn = sqlite3.connect("chatgpt_clone.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect("chatgpt_clone.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None

# Save chat message
def save_message(user_id, message, role):
    conn = sqlite3.connect("chatgpt_clone.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, message, role, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, message, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

# Load chat history
def load_chat_history(user_id):
    conn = sqlite3.connect("chatgpt_clone.db")
    cursor = conn.cursor()
    cursor.execute("SELECT message, role, timestamp FROM chat_history WHERE user_id = ? ORDER BY id ASC", (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history

# Initialize database
init_db()

# Streamlit app
st.set_page_config(page_title="ChatGPT Clone", page_icon="🤖", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.title("Welcome to ChatGPT Clone")
    
    auth_choice = st.radio("Choose an option:", ["Login", "Sign Up"])

    if auth_choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
    elif auth_choice == "Sign Up":
        username = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password")
        if st.button("Sign Up"):
            if register_user(username, password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already taken. Please choose another.")
else:
    st.title("ChatGPT Clone")
    st.sidebar.button("Log Out", on_click=lambda: st.session_state.clear())
    st.sidebar.button("New Chat", on_click=lambda: st.session_state.pop("messages", None))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load chat history
    history = load_chat_history(st.session_state.user_id)
    for message, role, timestamp in history:
        if role == "user":
            st.chat_message("user").markdown(f"{message} ({timestamp})")
        elif role == "assistant":
            st.chat_message("assistant").markdown(f"{message} ({timestamp})")

    # Chat input
    user_message = st.chat_input("Type your message...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        save_message(st.session_state.user_id, user_message, "user")
        st.chat_message("user").markdown(user_message)

        # Placeholder response from the assistant
        assistant_message = "This is a placeholder response."
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        save_message(st.session_state.user_id, assistant_message, "assistant")
        st.chat_message("assistant").markdown(assistant_message)
