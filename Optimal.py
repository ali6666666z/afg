import streamlit as st
import sqlite3
import hashlib
import re
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure database directory exists
os.makedirs('database', exist_ok=True)

# Enhanced Database Initialization
def init_db():
    try:
        conn = sqlite3.connect('database/users.db')
        c = conn.cursor()
        
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      username TEXT UNIQUE, 
                      email TEXT UNIQUE,
                      password TEXT, 
                      created_at DATETIME)''')
        
        # Create login attempts table
        c.execute('''CREATE TABLE IF NOT EXISTS login_attempts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT,
                      attempt_time DATETIME,
                      status TEXT)''')
        
        conn.commit()
        logging.info("Database initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        st.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

# Enhanced Password Hashing
def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# Password Strength Validation
def validate_password(password):
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
    try:
        conn = sqlite3.connect('database/users.db')
        c = conn.cursor()
        
        # Check if username or email already exists
        c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
        if c.fetchone():
            logging.warning(f"Registration attempt with existing username or email: {username}, {email}")
            return False
        
        # Hash password and insert user
        hashed_password = hash_password(password)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)", 
                  (username, email, hashed_password, current_time))
        conn.commit()
        logging.info(f"User registered: {username}")
        return True
    except sqlite3.Error as e:
        logging.error(f"User registration error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# User Authentication
def authenticate_user(username, password):
    try:
        conn = sqlite3.connect('database/users.db')
        c = conn.cursor()
        hashed_password = hash_password(password)
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                  (username, hashed_password))
        user = c.fetchone()
        logging.info(f"Authentication attempt for {username}: {'Successful' if user else 'Failed'}")
        return user is not None
    except sqlite3.Error as e:
        logging.error(f"Authentication error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Streamlit Page Configuration
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="🤖",
    layout="centered"
)

# Login/Signup Interface (Previous implementation remains the same)
def login_signup():
    # ... (previous CSS and UI code remains the same)
    pass

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
