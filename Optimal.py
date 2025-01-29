# تهيئة المتغيرات الأساسية
import streamlit as st
import os
from importlib import import_module
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from streamlit_mic_recorder import speech_to_text
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تهيئة مفاتيح API
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# تهيئة حالة الجلسة
if "interface_language" not in st.session_state:
    st.session_state.interface_language = "English"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=False)

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# تحديث إعدادات الصفحة
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إضافة CSS مخصص
st.markdown("""
    <style>
        /* الألوان الرئيسية */
        :root {
            --bgc-blue: #0066B3;
            --bgc-light-blue: #00A0DC;
            --bgc-dark: #1A1A1A;
            --bgc-light: #FFFFFF;
            --bgc-gray: #F7F7F8;
            --bgc-border: #E5E5E5;
        }

        /* تنسيق الصفحة الرئيسية */
        .stApp {
            background-color: var(--bgc-gray);
        }

        /* تنسيق الشريط الجانبي */
        .css-1d391kg {
            background-color: var(--bgc-dark);
        }

        /* تنسيق محادثات ChatGPT */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }

        .message-container {
            display: flex;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-bottom: 1px solid var(--bgc-border);
        }

        .user-message {
            background-color: white;
        }

        .assistant-message {
            background-color: var(--bgc-gray);
        }

        .message-avatar {
            width: 30px;
            height: 30px;
            margin-right: 1rem;
            border-radius: 2px;
        }

        .message-content {
            flex: 1;
            line-height: 1.6;
        }

        /* تنسيق مربع الإدخال */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid var(--bgc-border);
            z-index: 1000;
        }

        .stChatInput {
            max-width: 800px;
            margin: 0 auto;
            border: 1px solid var(--bgc-border);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        /* تنسيق الأزرار */
        .stButton>button {
            background-color: var(--bgc-blue);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: var(--bgc-light-blue);
        }

        /* تنسيق الروابط */
        a {
            color: var(--bgc-blue);
            text-decoration: none;
        }
        a:hover {
            color: var(--bgc-light-blue);
        }

        /* تنسيق البطاقات */
        .info-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# تحديث واجهة الصفحة الرئيسية
def render_main_header():
    st.markdown("""
        <div class="main-header">
            <div style="text-align: center;">
                <img src="BGC Logo Colored.svg" style="width: 150px; margin-bottom: 1rem;">
            </div>
            <h1 style="text-align: center; color: white;">
                POWERING PROGRESS IN IRAQ
            </h1>
            <p style="text-align: center; color: white; font-size: 1.2rem;">
                {subtitle}
            </p>
        </div>
    """.format(
        subtitle="شركة غاز البصرة - المساعد الذكي" if st.session_state.interface_language == "العربية" else "Basrah Gas Company - Intelligent Assistant"
    ), unsafe_allow_html=True)

# تحديث واجهة الشريط الجانبي
def render_sidebar():
    with st.sidebar:
        st.image("BGC Logo Colored.svg", width=150)
        st.markdown("---")
        
        # قسم اللغة
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.session_state.interface_language = st.selectbox(
            "Language | اللغة",
            ["English", "العربية"],
            index=0 if st.session_state.interface_language == "English" else 1
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # قسم الإعدادات
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        if st.session_state.interface_language == "العربية":
            st.markdown("### الإعدادات")
        else:
            st.markdown("### Settings")
        toggle_dark_mode()
        st.markdown('</div>', unsafe_allow_html=True)

# تحديث عرض الرسائل
def display_message(message):
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    avatar_src = "user-avatar.png" if message["role"] == "user" else "BGC Logo Colored.svg"
    
    st.markdown(f"""
        <div class="message-container {role_class}">
            <img src="{avatar_src}" class="message-avatar" alt="{message['role']}">
            <div class="message-content">
                {message["content"]}
            </div>
        </div>
    """, unsafe_allow_html=True)

# تعريف كلاس PDFSearchAndDisplay
class PDFSearchAndDisplay:
    def __init__(self):
        """تهيئة الكلاس"""
        self.fitz = __import__('fitz')  # استيراد PyMuPDF
        
    def capture_screenshots(self, pdf_path, highlighted_pages):
        """التقاط صور للصفحات المحددة من ملف PDF
        
        Args:
            pdf_path (str): مسار ملف PDF
            highlighted_pages (list): قائمة من أرقام الصفحات والنصوص المراد تمييزها
            
        Returns:
            list: قائمة من الصور الملتقطة
        """
        screenshots = []
        try:
            # فتح ملف PDF
            doc = self.fitz.open(pdf_path)
            
            # معالجة كل صفحة محددة
            for page_num, highlight_text in highlighted_pages:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    
                    # تحويل الصفحة إلى صورة
                    pix = page.get_pixmap(matrix=self.fitz.Matrix(2, 2))
                    
                    # تحويل الصورة إلى بايتس
                    img_bytes = pix.tobytes()
                    
                    # إضافة الصورة إلى القائمة
                    screenshots.append(img_bytes)
            
            # إغلاق الملف
            doc.close()
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة ملف PDF: {str(e)}")
            
        return screenshots

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

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

# Validate API key inputs and initialize components if valid
if groq_api_key and google_api_key:
    # Set Google API key as environment variable
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Initialize ChatGroq with the provided Groq API key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    # Define the chat prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful assistant for Basrah Gas Company (BGC). Your task is to answer questions based on the provided context about BGC. Follow these rules strictly:

        1. **Language Handling:**
           - If the question is in English, answer in English.
           - If the question is in Arabic, answer in Arabic.
           - If the user explicitly asks for a response in a specific language, respond in that language.

        2. **Contextual Answers:**
           - Provide accurate and concise answers based on the context provided.
           - Do not explicitly mention the source of information unless asked.

        3. **Handling Unclear or Unanswerable Questions:**
           - If the question is unclear or lacks sufficient context, respond with:
             - In English: "I'm sorry, I couldn't understand your question. Could you please provide more details?"
             - In Arabic: "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك تقديم المزيد من التفاصيل؟"
           - If the question cannot be answered based on the provided context, respond with:
             - In English: "I'm sorry, I don't have enough information to answer that question."
             - In Arabic: "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."

        4. **User Interface Language:**
           - If the user has selected Arabic as the interface language, prioritize Arabic in your responses unless the question is explicitly in English.
           - If the user has selected English as the interface language, prioritize English in your responses unless the question is explicitly in Arabic.

        5. **Professional Tone:**
           - Maintain a professional and respectful tone in all responses.
           - Avoid making assumptions or providing speculative answers.
        """),
        MessagesPlaceholder(variable_name="history"),  # Add chat history to the prompt
        ("human", "{input}"),
        ("system", "Context: {context}"),
    ])

    # Load existing embeddings from files
    if "vectors" not in st.session_state:
        with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if st.session_state.interface_language == "العربية" else "Loading embeddings... Please wait."):
            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )

            # Load existing FAISS index with safe deserialization
            embeddings_path = "embeddings"  # Path to your embeddings folder
            try:
                st.session_state.vectors = FAISS.load_local(
                    embeddings_path,
                    embeddings,
                    allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
                )
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحميل التضميدات: {str(e)}" if st.session_state.interface_language == "العربية" else f"Error loading embeddings: {str(e)}")
                st.session_state.vectors = None

    # Microphone button in the sidebar
    st.markdown("### الإدخال الصوتي" if st.session_state.interface_language == "العربية" else "### Voice Input")
    input_lang_code = "ar" if st.session_state.interface_language == "العربية" else "en"
    voice_input = speech_to_text(
        start_prompt="🎤",
        stop_prompt="⏹️ إيقاف" if st.session_state.interface_language == "العربية" else "⏹️ Stop",
        language=input_lang_code,
        use_container_width=True,
        just_once=True,
        key="mic_button",
    )

    # Reset button in the sidebar
    if st.button("إعادة تعيين الدردشة" if st.session_state.interface_language == "العربية" else "Reset Chat"):
        st.session_state.messages = []  # Clear chat history
        st.session_state.memory.clear()  # Clear memory
        st.success("تمت إعادة تعيين الدردشة بنجاح." if st.session_state.interface_language == "العربية" else "Chat has been reset successfully.")
        st.rerun()  # Rerun the app to reflect changes immediately
else:
    st.error("الرجاء إدخال مفاتيح API للمتابعة." if st.session_state.interface_language == "العربية" else "Please enter both API keys to proceed.")

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=False)

# List of negative phrases to check for unclear or insufficient answers
negative_phrases = [
    "I'm sorry",
    "عذرًا",
    "لا أملك معلومات كافية",
    "I don't have enough information",
    "لم أتمكن من فهم سؤالك",
    "I couldn't understand your question",
    "لا يمكنني الإجابة على هذا السؤال",
    "I cannot answer this question",
    "يرجى تقديم المزيد من التفاصيل",
    "Please provide more details",
    "غير واضح",
    "Unclear",
    "غير متأكد",
    "Not sure",
    "لا أعرف",
    "I don't know",
    "غير متاح",
    "Not available",
    "غير موجود",
    "Not found",
    "غير معروف",
    "Unknown",
    "غير محدد",
    "Unspecified",
    "غير مؤكد",
    "Uncertain",
    "غير كافي",
    "Insufficient",
    "غير دقيق",
    "Inaccurate",
    "غير مفهوم",
    "Not clear",
    "غير مكتمل",
    "Incomplete",
    "غير صحيح",
    "Incorrect",
    "غير مناسب",
    "Inappropriate",
    "Please provide me",
    "يرجى تزويدي",
    "Can you provide more",
    "هل يمكنك تقديم المزيد"
]

# Function to display response with references and screenshots
def display_response_with_references(response, assistant_response):
    """عرض رد المساعد مع المراجع والصور"""
    with st.chat_message("assistant"):
        # عرض رد المساعد
        st.markdown(assistant_response)
        
        # عرض المراجع والصور إذا لم يكن الرد يحتوي على عبارات سلبية
        if not any(phrase in assistant_response for phrase in negative_phrases):
            st.markdown("---")  # خط فاصل
            if "context" in response:
                # استخراج أرقام الصفحات الفريقة من السياق
                page_numbers = set()
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "unknown")
                    if page_number != "unknown" and str(page_number).isdigit():
                        page_numbers.add(int(page_number))

                # عرض أرقام الصفحات
                if page_numbers:
                    sorted_pages = sorted(page_numbers)
                    page_numbers_str = ", ".join(map(str, sorted_pages))
                    st.markdown(
                        f"**{'المصدر' if st.session_state.interface_language == 'العربية' else 'Source'}:** " +
                        f"{'صفحة رقم' if st.session_state.interface_language == 'العربية' else 'Page'} {page_numbers_str}"
                    )

                    # التقاط وعرض لقطات الشاشة للصفحات ذات الصلة
                    screenshots = []
                    for page_num in sorted_pages:
                        highlighted_pages = [(page_num, "")]
                        page_screenshots = pdf_searcher.capture_screenshots(pdf_path, highlighted_pages)
                        screenshots.extend(page_screenshots)
                    
                    # عرض الصور في شبكة
                    if screenshots:
                        cols = st.columns(min(len(screenshots), 2))  # عرض صورتين كحد أقصى في كل صف
                        for idx, (screenshot, page_num) in enumerate(zip(screenshots, sorted_pages)):
                            with cols[idx % 2]:
                                st.image(
                                    screenshot,
                                    caption=f"{'صفحة' if st.session_state.interface_language == 'العربية' else 'Page'} {page_num}",
                                    use_container_width=True
                                )

# Display chat history
for message in st.session_state.messages:
    display_message(message)

# If voice input is detected, process it
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    display_message({"role": "user", "content": voice_input})

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({
            "input": voice_input,
            "context": retriever.get_relevant_documents(voice_input),
            "history": st.session_state.memory.chat_memory.messages  # Include chat history
        })
        assistant_response = response["answer"]

        # Use the new function to display the response with references
        display_response_with_references(response, assistant_response)
        
        # Save messages to memory
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.memory.chat_memory.add_user_message(voice_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

# Text input field
if st.session_state.interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

# If text input is detected, process it
if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    display_message({"role": "user", "content": human_input})

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({
            "input": human_input,
            "context": retriever.get_relevant_documents(human_input),
            "history": st.session_state.memory.chat_memory.messages  # Include chat history
        })
        assistant_response = response["answer"]

        # Use the new function to display the response with references
        display_response_with_references(response, assistant_response)
        
        # Save messages to memory
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

# دالة لعرض واجهة تسجيل الدخول
def render_auth_interface():
    st.markdown("""
        <div class="auth-container">
            <div class="info-card">
                <h2 style="text-align: center;">
                    {title}
                </h2>
                <p style="text-align: center;">
                    {subtitle}
                </p>
            </div>
        </div>
    """.format(
        title="تسجيل الدخول" if st.session_state.interface_language == "العربية" else "Login",
        subtitle="الرجاء تسجيل الدخول للمتابعة" if st.session_state.interface_language == "العربية" else "Please login to continue"
    ), unsafe_allow_html=True)

    # نموذج تسجيل الدخول
    with st.form("login_form"):
        email = st.text_input(
            "البريد الإلكتروني" if st.session_state.interface_language == "العربية" else "Email",
            key="email"
        )
        password = st.text_input(
            "كلمة المرور" if st.session_state.interface_language == "العربية" else "Password",
            type="password",
            key="password"
        )
        
        # زر تسجيل الدخول
        submit = st.form_submit_button(
            "تسجيل الدخول" if st.session_state.interface_language == "العربية" else "Login"
        )
        
        if submit:
            # هنا يمكنك إضافة منطق التحقق من صحة بيانات تسجيل الدخول
            # للتجربة، سنقوم بتسجيل الدخول مباشرة
            st.session_state.is_authenticated = True
            st.success(
                "تم تسجيل الدخول بنجاح!" if st.session_state.interface_language == "العربية" else "Successfully logged in!"
            )
            st.rerun()

    # رابط إنشاء حساب جديد
    st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <p>
                {text} <a href="#">{link_text}</a>
            </p>
        </div>
    """.format(
        text="ليس لديك حساب؟" if st.session_state.interface_language == "العربية" else "Don't have an account?",
        link_text="إنشاء حساب جديد" if st.session_state.interface_language == "العربية" else "Create new account"
    ), unsafe_allow_html=True)

# دالة للتحكم في الوضع الداكن
def toggle_dark_mode():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    
    dark_mode = st.checkbox(
        "الوضع الداكن" if st.session_state.interface_language == "العربية" else "Dark Mode",
        value=st.session_state.dark_mode,
        key="dark_mode_toggle"
    )
    
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        if dark_mode:
            st.markdown("""
                <style>
                    :root {
                        --bgc-blue: #00A0DC;
                        --bgc-light-blue: #33B5E5;
                        --bgc-dark: #1A1A1A;
                        --bgc-light: #2D2D2D;
                    }
                    
                    .stApp {
                        background-color: var(--bgc-dark);
                        color: white;
                    }
                    
                    .info-card {
                        background-color: var(--bgc-light);
                        color: white;
                    }
                    
                    .stButton>button {
                        background-color: var(--bgc-blue);
                        color: white;
                    }
                    
                    .stChatMessage {
                        background-color: var(--bgc-light);
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <style>
                    :root {
                        --bgc-blue: #0066B3;
                        --bgc-light-blue: #00A0DC;
                        --bgc-dark: #1A1A1A;
                        --bgc-light: #FFFFFF;
                    }
                    
                    .stApp {
                        background-color: white;
                        color: black;
                    }
                    
                    .info-card {
                        background-color: white;
                        color: black;
                    }
                    
                    .stButton>button {
                        background-color: var(--bgc-blue);
                        color: white;
                    }
                    
                    .stChatMessage {
                        background-color: white;
                        color: black;
                    }
                </style>
            """, unsafe_allow_html=True)

# تحديث الدالة الرئيسية
def main():
    if not st.session_state.is_authenticated:
        render_auth_interface()
    else:
        render_main_header()
        
        # عرض المحادثة
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            display_message(message)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # مربع الإدخال
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        human_input = st.chat_input(
            "اكتب سؤالك هنا..." if st.session_state.interface_language == "العربية" else "Type your question here..."
        )
        
        # زر المايكروفون
        col1, col2 = st.columns([6, 1])
        with col2:
            voice_input = speech_to_text(
                "🎤",
                "⏹️",
                language="ar" if st.session_state.interface_language == "العربية" else "en",
                just_once=True,
                key="voice_input"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # معالجة المدخلات
        if human_input or voice_input:
            process_input(human_input or voice_input)

if __name__ == "__main__":
    main()
