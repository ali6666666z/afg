import streamlit as st
import sqlite3
from datetime import datetime
import hashlib

# --- Initialize SQLite Database ---
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_user(email, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, hashed_password))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(username, email, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                       (username, email, hashed_password))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

# --- Streamlit UI ---
st.set_page_config(page_title="Modern Login App", page_icon="🔒", layout="centered")
init_db()

# --- State Management ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Sign In ---
def sign_in():
    st.title("🔑 Sign In")
    with st.form(key="signin_form"):
        email = st.text_input("Email Address", placeholder="Enter your email", key="signin_email")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="signin_password")
        submit = st.form_submit_button("Sign In")

        if submit:
            user = validate_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.success(f"Welcome back, {user[1]}! 🎉")
            else:
                st.error("Invalid email or password. Please try again.")

# --- Sign Up ---
def sign_up():
    st.title("📝 Sign Up")
    with st.form(key="signup_form"):
        username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
        email = st.text_input("Email Address", placeholder="Enter your email", key="signup_email")
        password = st.text_input("Password", placeholder="Enter a password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password", key="signup_confirm_password")
        submit = st.form_submit_button("Sign Up")

        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif create_user(username, email, password):
                st.success("Account created successfully! Please sign in.")
            else:
                st.error("Username or email already exists. Please try again.")

# --- Main App ---
if st.session_state.logged_in:
    st.title(f"Welcome, {st.session_state.username}! 👋")
    st.write("You are now logged in.")
    if st.button("Sign Out"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.info("You have been signed out.")
else:
    st.sidebar.title("Welcome! 👋")
    page = st.sidebar.radio("Navigation", ["Sign In", "Sign Up"])
    if page == "Sign In":
        sign_in()
    else:
        sign_up()
