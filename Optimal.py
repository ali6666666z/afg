import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text  # Import speech-to-text function
import fitz  # PyMuPDF for capturing screenshots
import pdfplumber  # For searching text in PDF

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

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
        /* الألوان الرئيسية لشركة غاز البصرة */
        :root {
            --bgc-blue: #0066B3;
            --bgc-light-blue: #00A0DC;
            --bgc-dark: #1A1A1A;
            --bgc-light: #FFFFFF;
        }

        /* تنسيق العنوان الرئيسي */
        .main-header {
            background: linear-gradient(90deg, var(--bgc-blue) 0%, var(--bgc-light-blue) 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }

        /* تنسيق الشريط الجانبي */
        .css-1d391kg {
            background-color: var(--bgc-dark);
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

        /* تنسيق مربع الدردشة */
        .stChatMessage {
            border-radius: 10px;
            margin: 0.5rem 0;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        /* تنسيق مربع الإدخال */
        .stChatInput {
            border: 2px solid var(--bgc-blue);
            border-radius: 5px;
            padding: 0.5rem;
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

        /* تنسيق الأيقونات */
        .icon-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 50px;
            height: 50px;
            background-color: var(--bgc-blue);
            border-radius: 50%;
            margin: 0 auto 1rem auto;
        }
    </style>
""", unsafe_allow_html=True)

# تحديث واجهة الصفحة الرئيسية
def render_main_header():
    st.markdown("""
        <div class="main-header">
            <h1>POWERING PROGRESS IN IRAQ</h1>
            <p>Basrah Gas Company ChatBot - Your Intelligent Assistant</p>
        </div>
    """, unsafe_allow_html=True)

# تحديث واجهة الشريط الجانبي
def render_sidebar():
    with st.sidebar:
        st.image("BGC Logo Colored.svg", width=150)
        st.markdown("---")
        
        # قسم اللغة
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        interface_language = st.selectbox(
            "Language | اللغة",
            ["English", "العربية"],
            index=0 if interface_language == "English" else 1
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # قسم الإعدادات
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        if interface_language == "العربية":
            st.markdown("### الإعدادات")
        else:
            st.markdown("### Settings")
        toggle_dark_mode()
        st.markdown('</div>', unsafe_allow_html=True)

# تحديث عرض الرسائل
def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="message-content">{message["content"]}</div>', unsafe_allow_html=True)

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

# PDF Search and Screenshot Class
class PDFSearchAndDisplay:
    def __init__(self):
        pass

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

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
        with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
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
                st.error(f"حدث خطأ أثناء تحميل التضميدات: {str(e)}" if interface_language == "العربية" else f"Error loading embeddings: {str(e)}")
                st.session_state.vectors = None

    # Microphone button in the sidebar
    st.markdown("### الإدخال الصوتي" if interface_language == "العربية" else "### Voice Input")
    input_lang_code = "ar" if interface_language == "العربية" else "en"  # Set language code based on interface language
    voice_input = speech_to_text(
        start_prompt="🎤",
        stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
        language=input_lang_code,  # Language (en for English, ar for Arabic)
        use_container_width=True,
        just_once=True,
        key="mic_button",
    )

    # Reset button in the sidebar
    if st.button("إعادة تعيين الدردشة" if interface_language == "العربية" else "Reset Chat"):
        st.session_state.messages = []  # Clear chat history
        st.session_state.memory.clear()  # Clear memory
        st.success("تمت إعادة تعيين الدردشة بنجاح." if interface_language == "العربية" else "Chat has been reset successfully.")
        st.rerun()  # Rerun the app to reflect changes immediately
else:
    st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

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
    "Please provide me",  # إضافة هذه العبارة
    "يرجى تزويدي",  # إضافة هذه العبارة
    "Can you provide more",  # إضافة هذه العبارة
    "هل يمكنك تقديم المزيد"  # إضافة هذه العبارة
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
                        f"**{'المصدر' if interface_language == 'العربية' else 'Source'}:** " +
                        f"{'صفحة رقم' if interface_language == 'العربية' else 'Page'} {page_numbers_str}"
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
                                    caption=f"{'صفحة' if interface_language == 'العربية' else 'Page'} {page_num}",
                                    use_container_width=True
                                )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If voice input is detected, process it
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
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
if interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

# If text input is detected, process it
if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
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
            "history": st.session_state.memory.chat_memory.messages  # Include chat history
        })
        assistant_response = response["answer"]

        # Use the new function to display the response with references
        display_response_with_references(response, assistant_response)
        
        # Save messages to memory
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

# تحديث الدالة الرئيسية
def main():
    render_main_header()
    render_sidebar()
    
    # عرض المحتوى الرئيسي
    if not st.session_state.is_authenticated:
        render_auth_interface()
    else:
        # عرض واجهة الدردشة
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                display_message(message)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            if interface_language == "العربية":
                st.markdown("### معلومات سريعة")
                st.markdown("- اطرح أسئلتك باللغة العربية أو الإنجليزية")
                st.markdown("- استخدم الميكروفون للإدخال الصوتي")
                st.markdown("- اضغط على زر المحادثة الجديدة لبدء محادثة جديدة")
            else:
                st.markdown("### Quick Info")
                st.markdown("- Ask questions in Arabic or English")
                st.markdown("- Use the microphone for voice input")
                st.markdown("- Click New Chat to start a fresh conversation")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()س
