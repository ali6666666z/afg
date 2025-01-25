import streamlit as st
import os
import sqlite3
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, content TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

# Function to add a new user to the database
def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

# Function to authenticate a user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Function to save chat history
def save_chat_history(username, role, content):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO chat_history (username, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (username, role, content, timestamp))
    conn.commit()
    conn.close()

# Function to load chat history for a specific user
def load_chat_history(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chat_history WHERE username = ? ORDER BY timestamp", (username,))
    history = c.fetchall()
    conn.close()
    return history

# Function to get the last conversation timestamp for a user
def get_last_conversation_timestamp(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT timestamp FROM chat_history WHERE username = ? ORDER BY timestamp DESC LIMIT 1", (username,))
    last_timestamp = c.fetchone()
    conn.close()
    return last_timestamp[0] if last_timestamp else None

# Initialize the database
init_db()

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Login and Signup interface
def login_signup():
    st.title("BGC ChatBot")
    st.write("Welcome to the BGC ChatBot. Please login or signup to continue.")
    
    menu = st.selectbox("Menu", ["Login", "Signup"])
    
    if menu == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
    
    elif menu == "Signup":
        st.subheader("Signup")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Signup"):
            if new_password == confirm_password:
                try:
                    add_user(new_username, new_password)
                    st.success("Account created successfully! Please login.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists.")
            else:
                st.error("Passwords do not match")

# Main chat interface
def chat_interface():
    # Sidebar configuration
    with st.sidebar:
        st.title("المحادثات السابقة" if st.session_state.interface_language == "العربية" else "Previous Chats")
        
        # Load all users (for demonstration purposes)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT username FROM users")
        users = c.fetchall()
        conn.close()
        
        # Display previous chats with last conversation timestamp
        for user in users:
            username = user[0]
            last_timestamp = get_last_conversation_timestamp(username)
            if last_timestamp:
                st.write(f"**{username}** - Last chat: {last_timestamp}")
            else:
                st.write(f"**{username}** - No chats yet")

    # Main area for chat interface
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image("BGC Logo Colored.svg", width=100)

    with col2:
        if st.session_state.interface_language == "العربية":
            st.title("محمد الياسين | بوت الدردشة BGC")
            st.write("""
            **مرحبًا!**  
            هذا بوت الدردشة الخاص بشركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.  
            **كيفية الاستخدام:**  
            - اكتب سؤالك في مربع النص أدناه.  
            - أو استخدم زر المايكروفون للتحدث مباشرة.  
            - سيتم الرد عليك بناءً على المعلومات المتاحة.  
            """)
        else:
            st.title("Mohammed Al-Yaseen | BGC ChatBot")
            st.write("""
            **Welcome!**  
            This is the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.  
            **How to use:**  
            - Type your question in the text box below.  
            - Or use the microphone button to speak directly.  
            - You will receive a response based on the available information.  
            """)

    # Initialize session state for chat messages if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history(st.session_state.username)

    # Initialize memory if not already done
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # If voice input is detected, process it
    if voice_input:
        st.session_state.messages.append({"role": "user", "content": voice_input})
        save_chat_history(st.session_state.username, "user", voice_input)
        with st.chat_message("user"):
            st.markdown(voice_input)

        if "vectors" in st.session_state and st.session_state.vectors is not None:
            # Create and configure the document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Get response from the assistant
            response = retrieval_chain.invoke({
                "input": voice_input,
                "context": retriever.get_relevant_documents(voice_input),
                "history": st.session_state.memory.chat_memory.messages
            })
            assistant_response = response["answer"]

            # Append and display assistant's response
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            save_chat_history(st.session_state.username, "assistant", assistant_response)
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Add user and assistant messages to memory
            st.session_state.memory.chat_memory.add_user_message(voice_input)
            st.session_state.memory.chat_memory.add_ai_message(assistant_response)

            # Display supporting information (page numbers only)
            with st.expander("المعلومات الداعمة" if st.session_state.interface_language == "العربية" else "Supporting Information"):
                if "context" in response:
                    # Extract unique page numbers from the context
                    page_numbers = set()
                    for doc in response["context"]:
                        page_number = doc.metadata.get("page", "unknown")
                        if page_number != "unknown" and str(page_number).isdigit():
                            page_numbers.add(int(page_number))

                    # Display the page numbers
                    if page_numbers:
                        page_numbers_str = ", ".join(map(str, sorted(page_numbers)))
                        st.write(f"هذه الإجابة وفقًا للصفحات: {page_numbers_str}" if st.session_state.interface_language == "العربية" else f"This answer is according to pages: {page_numbers_str}")
                    else:
                        st.write("لا توجد أرقام صفحات صالحة في السياق." if st.session_state.interface_language == "العربية" else "No valid page numbers available in the context.")
                else:
                    st.write("لا يوجد سياق متاح." if st.session_state.interface_language == "العربية" else "No context available.")
        else:
            # Prompt user to ensure embeddings are loaded
            assistant_response = (
                "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if st.session_state.interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            save_chat_history(st.session_state.username, "assistant", assistant_response)
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

    # Text input field
    if st.session_state.interface_language == "العربية":
        human_input = st.chat_input("اكتب سؤالك هنا...")
    else:
        human_input = st.chat_input("Type your question here...")

    # If text input is detected, process it
    if human_input:
        st.session_state.messages.append({"role": "user", "content": human_input})
        save_chat_history(st.session_state.username, "user", human_input)
        with st.chat_message("user"):
            st.markdown(human_input)

        if "vectors" in st.session_state and st.session_state.vectors is not None:
            # Create and configure the document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Get response from the assistant
            response = retrieval_chain.invoke({
                "input": human_input,
                "context": retriever.get_relevant_documents(human_input),
                "history": st.session_state.memory.chat_memory.messages
            })
            assistant_response = response["answer"]

            # Append and display assistant's response
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            save_chat_history(st.session_state.username, "assistant", assistant_response)
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Add user and assistant messages to memory
            st.session_state.memory.chat_memory.add_user_message(human_input)
            st.session_state.memory.chat_memory.add_ai_message(assistant_response)

            # Display supporting information (page numbers only)
            with st.expander("مراجع الصفحات" if st.session_state.interface_language == "العربية" else "Page References"):
                if "context" in response:
                    # Extract unique page numbers from the context
                    page_numbers = set()
                    for doc in response["context"]:
                        page_number = doc.metadata.get("page", "unknown")
                        if page_number != "unknown" and str(page_number).isdigit():
                            page_numbers.add(int(page_number))

                    # Display the page numbers
                    if page_numbers:
                        page_numbers_str = ", ".join(map(str, sorted(page_numbers)))
                        st.write(f"هذه الإجابة وفقًا للصفحات: {page_numbers_str}" if st.session_state.interface_language == "العربية" else f"This answer is according to pages: {page_numbers_str}")
                    else:
                        st.write("لا توجد أرقام صفحات صالحة في السياق." if st.session_state.interface_language == "العربية" else "No valid page numbers available in the context.")
                else:
                    st.write("لا يوجد سياق متاح." if st.session_state.interface_language == "العربية" else "No context available.")
        else:
            # Prompt user to ensure embeddings are loaded
            assistant_response = (
                "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if st.session_state.interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            save_chat_history(st.session_state.username, "assistant", assistant_response)
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

# Main app logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    chat_interface()
else:
    login_signup()
