import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import speech_to_text  # Import speech-to-text function
import fitz  # PyMuPDF for capturing screenshots
import pdfplumber  # For searching text in PDF
import re

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
        """تهيئة الكلاس"""
        self.fitz = fitz  # استخدام fitz المستورد مباشرة
        
    def get_text_instances(self, page_text, search_text):
        """البحث عن جميع مواضع النص في الصفحة"""
        instances = []
        search_text = search_text.lower()
        page_text = page_text.lower()
        
        start = 0
        while True:
            index = page_text.find(search_text, start)
            if index == -1:
                break
            instances.append((index, index + len(search_text)))
            start = index + 1
            
        return instances

    def capture_screenshots(self, pdf_path, page_info):
        """التقاط صور للصفحات المحددة من ملف PDF مع تمييز النص
        
        Args:
            pdf_path (str): مسار ملف PDF
            page_info (list): قائمة من (رقم الصفحة، النص المقتبس)
        """
        screenshots = []
        try:
            # فتح ملف PDF
            doc = self.fitz.open(pdf_path)
            
            # معالجة كل صفحة محددة
            for page_num, quoted_text in page_info:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    
                    # البحث عن النص المقتبس في الصفحة
                    if quoted_text:
                        # الحصول على النص الكامل للصفحة
                        page_text = page.get_text()
                        
                        # البحث عن جميع مواضع النص المقتبس
                        text_instances = self.get_text_instances(page_text, quoted_text)
                        
                        # تمييز كل موضع للنص
                        for start, end in text_instances:
                            # البحث عن مواضع النص على الصفحة
                            text_instances = page.search_for(quoted_text)
                            
                            # إضافة تمييز لكل موضع
                            for inst in text_instances:
                                highlight = page.add_highlight_annot(inst)
                                highlight.set_colors({"stroke": (1, 1, 0)})  # لون أصفر للتمييز
                                highlight.update()
                    
                    # تحويل الصفحة إلى صورة بدقة عالية
                    zoom = 2  # مضاعفة الدقة
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # تحويل الصورة إلى بايتس
                    img_bytes = pix.tobytes()
                    
                    # إضافة الصورة إلى القائمة
                    screenshots.append(img_bytes)
            
            # إغلاق الملف
            doc.close()
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة ملف PDF: {str(e)}")
            
        return screenshots

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_page_screenshot(self, page_num):
        """التقاط صورة للصفحة المحددة من ملف PDF"""
        try:
            # فتح ملف PDF
            doc = self.fitz.open("BGC.pdf")
            
            # معالجة الصفحة المحددة
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                
                # تحويل الصفحة إلى صورة بدقة عالية
                zoom = 2  # مضاعفة الدقة
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # تحويل الصورة إلى بايتس
                img_bytes = pix.tobytes()
                
                # إغلاق الملف
                doc.close()
                
                return img_bytes
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة ملف PDF: {str(e)}")
            
        return None

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

        # تعريف القالب الأساسي للدردشة
        def create_chat_prompt():
            return PromptTemplate(
                template="""You are an expert assistant for Basrah Gas Company (BGC). Your task is to provide accurate and detailed answers based on the available information. Follow these instructions carefully:

1. If the user asks in Arabic, respond in Arabic. If they ask in English, respond in English.
2. Provide direct and clear answers.
3. Use only the information from the provided context.
4. Stay professional and organized.

Context:
{context}

Question: {input}

Remember to:
1. Match the language of the question
2. Be precise and accurate
3. Stay organized and professional
""",
                input_variables=["context", "input"]
            )

        def create_custom_chain(llm, prompt):
            """إنشاء سلسلة معالجة المستندات"""
            return create_stuff_documents_chain(
                llm=llm,
                prompt=prompt
            )

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

def detect_language(text):
    """تحديد لغة النص تلقائياً"""
    # التحقق من وجود حروف عربية في النص
    arabic_pattern = re.compile('[\u0600-\u06FF]')
    if arabic_pattern.search(text):
        return "ar"
    return "en"

def get_relevant_context(retriever, query, k=5):
    """الحصول على السياق الأكثر صلة"""
    docs = retriever.get_relevant_documents(
        query,
        search_kwargs={"k": k}
    )
    return docs[:k]

def process_input(input_text, retriever, llm, memory):
    """معالجة إدخال المستخدم مع دعم اللغتين"""
    try:
        # تحديد لغة السؤال
        lang = detect_language(input_text)
        
        # الحصول على السياق
        context = get_relevant_context(retriever, input_text)
        
        # إنشاء القالب والسلسلة
        prompt = create_chat_prompt()
        chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=create_custom_chain(llm, prompt)
        )
        
        # إعداد السؤال مع تعليمات اللغة
        enhanced_input = input_text
        if lang == "ar":
            enhanced_input = f"أجب باللغة العربية: {input_text}"
        else:
            enhanced_input = f"Please answer in English: {input_text}"
        
        # الحصول على الإجابة
        response = chain.invoke({
            "input": enhanced_input,
            "history": memory.load_memory_variables({})["history"]
        })
        
        return {
            "answer": response["answer"],
            "context": context,
            "language": lang
        }
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة السؤال: {str(e)}")
        return None

def display_response_with_references(response_data):
    """عرض الإجابة مع الصور"""
    if not response_data:
        return
        
    # عرض الإجابة
    if isinstance(response_data, dict):
        if "answer" in response_data:
            st.write(response_data["answer"])
        elif "content" in response_data:
            st.write(response_data["content"])
        
        # عرض الصور
        context_data = None
        if "context" in response_data:
            context_data = response_data["context"]
        elif "references" in response_data and isinstance(response_data["references"], dict):
            context_data = response_data["references"].get("context")
        
        if context_data:
            # تجميع الصور
            images_data = []
            for doc in context_data:
                page_num = doc.metadata.get("page", "غير معروف")
                try:
                    # التقاط لقطة من الصفحة بدون تظليل
                    image = pdf_searcher.capture_page_screenshot(page_num)
                    if image:
                        images_data.append((image, page_num))
                except Exception as e:
                    continue
            
            # عرض الصور في شبكة
            if images_data:
                cols = st.columns(2)
                for idx, (image, page_num) in enumerate(images_data):
                    with cols[idx % 2]:
                        st.image(image)
                        # عرض رقم الصفحة بناءً على لغة السؤال
                        lang = response_data.get("language", "en")
                        if lang == "ar":
                            st.markdown(f"**صفحة {page_num}**")
                        else:
                            st.markdown(f"**Page {page_num}**")
    else:
        st.write(response_data)

# عرض سجل المحادثة
for message in st.session_state.messages:
    if message["role"] == "assistant" and "references" in message:
        display_response_with_references(message["references"])
    else:
        st.write(message["content"])

# حقل إدخال النص
if interface_language == "العربية":
    human_input = st.text_input("اكتب سؤالك هنا...")
else:
    human_input = st.text_input("Type your question here...")

# معالجة الإدخال النصي
if human_input:
    user_message = {"role": "user", "content": human_input}
    st.session_state.messages.append(user_message)
    st.write(user_message["content"])

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        try:
            # معالجة الإدخال مع التحسينات الجديدة
            response = process_input(
                human_input,
                st.session_state.vectors.as_retriever(),
                llm,
                st.session_state.memory
            )
            
            if response:
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "references": response
                }
                st.session_state.messages.append(assistant_message)
                display_response_with_references(response)
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")

# معالجة الإدخال الصوتي بنفس الطريقة
if voice_input:
    # تحويل الصوت إلى نص مع الكشف التلقائي عن اللغة
    try:
        text = speech_to_text(voice_input)
        if text:
            user_message = {"role": "user", "content": text}
            st.session_state.messages.append(user_message)
            st.write(user_message["content"])

            if "vectors" in st.session_state and st.session_state.vectors is not None:
                response = process_input(text, st.session_state.vectors, llm, st.session_state.memory)
                if response:
                    assistant_message = {
                        "role": "assistant",
                        "content": response["answer"],
                        "references": response
                    }
                    st.session_state.messages.append(assistant_message)
                    display_response_with_references(response)
    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")

st.title("Basrah Gas Company Assistant - مساعد شركة غاز البصرة")
