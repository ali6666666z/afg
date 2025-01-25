import streamlit as st
import sqlite3
import hashlib
import re
from datetime import datetime

# Enhanced Database Initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  email TEXT UNIQUE,
                  created_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS login_attempts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  attempt_time DATETIME,
                  status TEXT)''')
    conn.commit()
    conn.close()

# Enhanced Password Hashing
def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# Password Strength Validation
def validate_password(password):
    # At least 8 characters, one uppercase, one lowercase, one number, one special char
    return (len(password) >= 8 and 
            re.search(r'[A-Z]', password) and 
            re.search(r'[a-z]', password) and 
            re.search(r'\d', password) and 
            re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

# Email Validation
def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email)

# User Registration
def add_user(username, email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)", 
                  (username, email, hashed_password, current_time))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# User Authentication
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
              (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Streamlit Page Configuration
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="🤖",
    layout="centered"
)

# Modern Login/Signup Interface
def login_signup():
    # Custom CSS for modern, sleek design
    st.markdown("""
    <style>
    .login-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 30px;
        max-width: 400px;
        margin: auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0e8f6d;
    }
    .stTextInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #d1d5db;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Login Container
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    # Logo or Title
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("BGC Logo Colored.svg", width=150)
    
    st.markdown("### BGC ChatBot Login")
    
    # Toggle between Login and Signup
    login_mode = st.radio("", ["Login", "Sign Up"], horizontal=True)

    if login_mode == "Login":
        # Login Form
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("Sign In"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful!")
            else:
                st.error("Invalid username or password")

    else:
        # Signup Form
        new_username = st.text_input("Choose Username", placeholder="Create a unique username")
        email = st.text_input("Email", placeholder="Enter your email address")
        new_password = st.text_input("Create Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")
        
        if st.button("Create Account"):
            # Comprehensive validation
            if not new_username or len(new_username) < 3:
                st.error("Username must be at least 3 characters long")
            elif not validate_email(email):
                st.error("Please enter a valid email address")
            elif not validate_password(new_password):
                st.error("""Password must:
                - Be at least 8 characters long
                - Contain at least one uppercase letter
                - Contain at least one lowercase letter
                - Contain at least one number
                - Contain at least one special character""")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                # Attempt to add user
                if add_user(new_username, email, new_password):
                    st.success("Account created successfully! You can now log in.")
                else:
                    st.error("Username or email already exists")

    # Additional login options
    st.markdown("---")
    st.markdown("*Forgot password? Contact system administrator*")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main Application Logic
def main():
    # Initialize database
    init_db()

    # Login/Signup or Chat Interface
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_signup()
    else:
        # Placeholder for chat interface
        st.write("Welcome to BGC ChatBot!")
        if st.button("Logout"):
            st.session_state.logged_in = False

if __name__ == "__main__":
    main()
