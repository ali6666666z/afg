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
        self.fitz = fitz
        self.pdfplumber = pdfplumber

    def capture_screenshots(self, pdf_path, pages):
        """التقاط صور من صفحات PDF محددة"""
        screenshots = []
        try:
            doc = self.fitz.open(pdf_path)
            for page_num, _ in pages:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    # تحويل الصفحة إلى صورة بدقة عالية
                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    screenshots.append(pix.tobytes())
            doc.close()
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة ملف PDF: {str(e)}")
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

        # تعريف القالب الأساسي للدردشة
        def create_chat_prompt():
            return PromptTemplate(
                template="""أنت مساعد مفيد لشركة غاز البصرة (BGC). مهمتك هي الإجابة على الأسئلة بناءً على السياق المقدم حول BGC. اتبع هذه القواعد بدقة:

                1. قدم إجابات دقيقة ومباشرة
                2. استخدم فقط المعلومات من السياق المقدم
                3. إذا لم تكن متأكداً، قل ذلك بصراحة
                4. حافظ على لغة مهنية ومحترفة

                السياق المقدم:
                {context}

                السؤال: {input}

                تذكر أن تقدم إجابة:
                1. دقيقة ومستندة إلى الوثائق
                2. مباشرة وواضحة
                3. مهنية ومنظمة
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

def clean_text(text):
    """تنظيف النص من الأخطاء والفراغات الزائدة"""
    # إزالة الفراغات الزائدة
    text = ' '.join(text.split())
    # إزالة علامات التنسيق غير المرغوبة
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text

def extract_complete_sentences(text, max_length=200):
    """استخراج جمل كاملة من النص"""
    # تقسيم النص إلى جمل
    sentences = text.split('.')
    complete_text = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # التأكد من أن الجملة تبدأ بحرف كبير وتنتهي بنقطة
        if sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'):
            sentence += '.'
            
        # إضافة الجملة إذا كانت ضمن الحد الأقصى للطول
        if current_length + len(sentence) <= max_length:
            complete_text.append(sentence)
            current_length += len(sentence)
        else:
            break
            
    return ' '.join(complete_text)

def get_relevant_context(retriever, query, k=3):
    """الحصول على السياق الأكثر صلة وتنظيمه"""
    # استرجاع المستندات ذات الصلة
    docs = retriever.get_relevant_documents(query)
    
    # تنظيم وتنقية السياق
    organized_context = []
    for doc in docs[:k]:  # استخدام أفضل k مستندات فقط
        text = clean_text(doc.page_content)
        complete_text = extract_complete_sentences(text)
        if complete_text:
            # إنشاء وثيقة جديدة مع النص المنظم
            organized_doc = Document(
                page_content=complete_text,
                metadata={"page": doc.metadata.get("page", "unknown")}
            )
            organized_context.append(organized_doc)
    
    return organized_context

def process_input(input_text, retriever, llm, memory):
    """معالجة إدخال المستخدم مع تحسين جودة الإجابات والسياق"""
    try:
        # الحصول على السياق المنظم
        context = get_relevant_context(retriever, input_text)
        
        # إنشاء القالب والسلسلة
        prompt = create_chat_prompt()
        chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=create_custom_chain(llm, prompt)
        )
        
        # الحصول على الإجابة
        response = chain.invoke({
            "input": input_text,
            "history": memory.load_memory_variables({})["history"]
        })
        
        # تنظيم الإجابة والسياق
        organized_response = {
            "answer": response["answer"],
            "context": context
        }
        
        return organized_response
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة السؤال: {str(e)}")
        return None

def display_references(refs):
    if refs and "context" in refs:
        # استخراج أرقام الصفحات فقط
        page_info = []
        for doc in refs["context"]:
            page_number = doc.metadata.get("page", "unknown")
            if page_number != "unknown" and str(page_number).isdigit():
                page_info.append(int(page_number))

        # عرض الصور
        if page_info:
            # إنشاء عمودين لعرض الصور
            cols = st.columns(2)
            
            # التقاط وعرض لقطات الشاشة للصفحات
            for idx, page_num in enumerate(sorted(set(page_info))):
                col_idx = idx % 2
                with cols[col_idx]:
                    screenshots = pdf_searcher.capture_screenshots(pdf_path, [(page_num, "")])
                    if screenshots:
                        st.image(screenshots[0], use_container_width=True)
                        st.markdown(f"**صفحة {page_num}**")

def display_chat_message(message, with_refs=False):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if with_refs and "references" in message:
            display_references(message["references"])

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

# معالجة الإدخال النصي
if human_input:
    user_message = {"role": "user", "content": human_input}
    st.session_state.messages.append(user_message)
    display_chat_message(user_message)

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
                    "references": {"context": response["context"]}
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.memory.chat_memory.add_user_message(human_input)
                st.session_state.memory.chat_memory.add_ai_message(response["answer"])

                # عرض الرد مع المراجع والصور
                display_response_with_references(response, response["answer"])
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")

# معالجة الإدخال الصوتي بنفس الطريقة
if voice_input:
    user_message = {"role": "user", "content": voice_input}
    st.session_state.messages.append(user_message)
    display_chat_message(user_message)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        try:
            response = process_input(
                voice_input,
                st.session_state.vectors.as_retriever(),
                llm,
                st.session_state.memory
            )
            
            if response:
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "references": {"context": response["context"]}
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.memory.chat_memory.add_user_message(voice_input)
                st.session_state.memory.chat_memory.add_ai_message(response["answer"])

                # عرض الرد مع المراجع والصور
                display_response_with_references(response, response["answer"])
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
