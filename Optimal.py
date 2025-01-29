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

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  # Page title
    page_icon="BGC Logo Colored.svg",  # New page icon
    layout="wide"  # Page layout
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

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")  # Right-to-left for Arabic
        st.title("الإعدادات")  # Sidebar title in Arabic
    else:
        apply_css_direction("ltr")  # Left-to-right for English
        st.title("Settings")  # Sidebar title in English

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

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main area for chat interface
# Use columns to display logo and title side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Display the logo in the first column
with col1:
    st.image("BGC Logo Colored.svg", width=100)  # Adjust the width as needed

# Display the title and description in the second column
with col2:
    if interface_language == "العربية":
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

# تحديث دالة عرض المحادثة
def display_chat_message(message, with_refs=False):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if with_refs and "references" in message:
            display_references(message["references"])

# دالة عرض المراجع والصور
def display_references(refs):
    if refs and "context" in refs:
        # استخراج أرقام الصفحات الفريدة من السياق
        page_numbers = set()
        for doc in refs["context"]:
            page_number = doc.metadata.get("page", "unknown")
            if page_number != "unknown" and str(page_number).isdigit():
                page_numbers.add(int(page_number))

        # عرض أرقام الصفحات
        if page_numbers:
            sorted_pages = sorted(page_numbers)
            page_numbers_str = ", ".join(map(str, sorted_pages))
            st.markdown("---")
            st.markdown(
                f"**{'المصدر' if interface_language == 'العربية' else 'Source'}:** " +
                f"{'صفحة رقم' if interface_language == 'العربية' else 'Page'} {page_numbers_str}"
            )

            # إنشاء عمودين لعرض الصور
            cols = st.columns(2)  # عمودان فقط
            
            # التقاط وعرض لقطات الشاشة للصفحات ذات الصلة
            for idx, page_num in enumerate(sorted_pages):
                col_idx = idx % 2  # تحديد رقم العمود (0 أو 1)
                with cols[col_idx]:
                    highlighted_pages = [(page_num, "")]
                    screenshots = pdf_searcher.capture_screenshots(pdf_path, highlighted_pages)
                    if screenshots:
                        # عرض الصورة بحجم مناسب
                        st.image(
                            screenshots[0],
                            use_container_width=True,
                            width=300  # عرض أكبر قليلاً لأن لدينا عمودين فقط
                        )
                        # عرض رقم الصفحة أسفل الصورة
                        st.markdown(
                            f"<div style='text-align: center;'><p><strong>{'صفحة' if interface_language == 'العربية' else 'Page'} {page_num}</strong></p></div>",
                            unsafe_allow_html=True
                        )

# تحديث دالة عرض الرد مع المراجع
def display_response_with_references(response, assistant_response):
    if not any(phrase in assistant_response for phrase in negative_phrases):
        # إضافة المراجع إلى الرسالة
        message = {
            "role": "assistant",
            "content": assistant_response,
            "references": response
        }
        display_chat_message(message, with_refs=True)
    else:
        # إذا كان الرد يحتوي على عبارات سلبية، نعرض الرد فقط
        display_chat_message({
            "role": "assistant",
            "content": assistant_response
        })

# عرض سجل المحادثة
for message in st.session_state.messages:
    if message["role"] == "assistant" and "references" in message:
        display_chat_message(message, with_refs=True)
    else:
        display_chat_message(message)

# حقل إدخال النص
if interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

# معالجة الإدخال الصوتي
if voice_input:
    user_message = {"role": "user", "content": voice_input}
    st.session_state.messages.append(user_message)
    display_chat_message(user_message)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        retriever = st.session_state.vectors.as_retriever()
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        response = chain.invoke({
            "input": voice_input,
            "history": st.session_state.memory.load_memory_variables({})["history"]
        })
        assistant_response = response["answer"]

        # حفظ الرسائل في الذاكرة مع المراجع
        assistant_message = {
            "role": "assistant",
            "content": assistant_response,
            "references": response
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.memory.chat_memory.add_user_message(voice_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        # عرض الرد مع المراجع والصور
        display_response_with_references(response, assistant_response)

# معالجة الإدخال النصي
if human_input:
    user_message = {"role": "user", "content": human_input}
    st.session_state.messages.append(user_message)
    display_chat_message(user_message)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        retriever = st.session_state.vectors.as_retriever()
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        response = chain.invoke({
            "input": human_input,
            "history": st.session_state.memory.load_memory_variables({})["history"]
        })
        assistant_response = response["answer"]

        # حفظ الرسائل في الذاكرة مع المراجع
        assistant_message = {
            "role": "assistant",
            "content": assistant_response,
            "references": response
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        # عرض الرد مع المراجع والصور
        display_response_with_references(response, assistant_response)
