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
st.set_page_config(page_title="Welcome to My App", page_icon="🔒", layout="centered")
init_db()

# --- State Management ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.page = "home"
    st.session_state.username = ""

# --- Welcome Page ---
def welcome_page():
    st.image("https://via.placeholder.com/300x100.png?text=Company+Logo", use_column_width=False)
    st.title("مرحباً بك في تطبيقنا!")
    st.markdown("### اختر ما تريد:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("تسجيل الدخول"):
            st.session_state.page = "sign_in"
    with col2:
        if st.button("إنشاء حساب"):
            st.session_state.page = "sign_up"

# --- Sign In Page ---
def sign_in():
    st.title("🔑 تسجيل الدخول")
    with st.form(key="signin_form"):
        email = st.text_input("البريد الإلكتروني", placeholder="أدخل بريدك الإلكتروني")
        password = st.text_input("كلمة المرور", placeholder="أدخل كلمة المرور", type="password")
        submit = st.form_submit_button("تسجيل الدخول")

        if submit:
            user = validate_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.success(f"مرحباً بعودتك، {user[1]}! 🎉")
            else:
                st.error("بيانات الدخول غير صحيحة، حاول مرة أخرى.")

# --- Sign Up Page ---
def sign_up():
    st.title("📝 إنشاء حساب")
    with st.form(key="signup_form"):
        username = st.text_input("اسم المستخدم", placeholder="اختر اسم مستخدم")
        email = st.text_input("البريد الإلكتروني", placeholder="أدخل بريدك الإلكتروني")
        password = st.text_input("كلمة المرور", placeholder="أدخل كلمة المرور", type="password")
        confirm_password = st.text_input("تأكيد كلمة المرور", placeholder="أعد كتابة كلمة المرور", type="password")
        submit = st.form_submit_button("إنشاء الحساب")

        if submit:
            if password != confirm_password:
                st.error("كلمات المرور غير متطابقة!")
            elif create_user(username, email, password):
                st.success("تم إنشاء الحساب بنجاح! يمكنك الآن تسجيل الدخول.")
            else:
                st.error("اسم المستخدم أو البريد الإلكتروني موجود مسبقاً. حاول مرة أخرى.")

# --- Main App Logic ---
if st.session_state.logged_in:
    st.title(f"مرحباً، {st.session_state.username}! 👋")
    st.write("أنت الآن مسجل الدخول.")
    if st.button("تسجيل الخروج"):
        st.session_state.logged_in = False
        st.session_state.page = "home"
        st.info("تم تسجيل الخروج.")
else:
    if st.session_state.page == "home":
        welcome_page()
    elif st.session_state.page == "sign_in":
        sign_in()
    elif st.session_state.page == "sign_up":
        sign_up()
