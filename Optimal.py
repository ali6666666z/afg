# استيراد المكتبات اللازمة
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
from streamlit_mic_recorder import speech_to_text
import fitz  # PyMuPDF for capturing screenshots
import pdfplumber  # For searching text in PDF

# تعريف المتغيرات العامة
negative_phrases = [
    "لا أملك معلومات كافية",
    "لم أتمكن من فهم سؤالك",
    "عذراً، لا أستطيع",
    "I don't have enough information",
    "I couldn't understand your question",
    "I apologize, I cannot",
    "هل يمكنك تقديم المزيد"
]

# دوال معالجة النصوص
def clean_text(text):
    """تنظيف النص من الأخطاء والفراغات الزائدة"""
    text = ' '.join(text.split())
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text

def extract_complete_sentences(text, max_length=200):
    """استخراج جمل كاملة من النص"""
    sentences = text.split('.')
    complete_text = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'):
            sentence += '.'
            
        if current_length + len(sentence) <= max_length:
            complete_text.append(sentence)
            current_length += len(sentence)
        else:
            break
            
    return ' '.join(complete_text)

# كلاس معالجة ملفات PDF
class PDFSearchAndDisplay:
    def __init__(self):
        self.fitz = fitz
        
    def get_text_instances(self, page_text, search_text):
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
        screenshots = []
        try:
            doc = self.fitz.open(pdf_path)
            
            for page_num, quoted_text in page_info:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    
                    if quoted_text:
                        page_text = page.get_text()
                        text_instances = self.get_text_instances(page_text, quoted_text)
                        
                        for start, end in text_instances:
                            text_instances = page.search_for(quoted_text)
                            
                            for inst in text_instances:
                                highlight = page.add_highlight_annot(inst)
                                highlight.set_colors({"stroke": (1, 1, 0)})
                                highlight.update()
                    
                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes()
                    screenshots.append(img_bytes)
            
            doc.close()
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة ملف PDF: {str(e)}")
            
        return screenshots

# دوال معالجة المحادثة
def create_chat_prompt():
    """إنشاء قالب المحادثة"""
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

def get_relevant_context(retriever, query, k=3):
    """الحصول على السياق الأكثر صلة"""
    docs = retriever.get_relevant_documents(query)
    organized_context = []
    
    for doc in docs[:k]:
        text = clean_text(doc.page_content)
        complete_text = extract_complete_sentences(text)
        if complete_text:
            organized_doc = Document(
                page_content=complete_text,
                metadata={"page": doc.metadata.get("page", "unknown")}
            )
            organized_context.append(organized_doc)
    
    return organized_context

def create_stuff_documents_chain(llm, prompt):
    """إنشاء سلسلة معالجة المستندات"""
    return create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

def process_input(input_text, retriever, llm, memory):
    """معالجة إدخال المستخدم"""
    try:
        context = get_relevant_context(retriever, input_text)
        prompt = create_chat_prompt()
        chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=create_stuff_documents_chain(llm, prompt)
        )
        
        response = chain.invoke({
            "input": input_text,
            "history": memory.load_memory_variables({})["history"]
        })
        
        return {
            "answer": response["answer"],
            "context": context
        }
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة السؤال: {str(e)}")
        return None

# دوال عرض المحادثة
def display_chat_message(message, with_refs=False):
    """عرض رسالة المحادثة"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if with_refs and "references" in message:
            display_references(message["references"])

def display_references(refs):
    """عرض المراجع والصور"""
    if refs and "context" in refs:
        page_info = []
        for doc in refs["context"]:
            page_number = doc.metadata.get("page", "unknown")
            if page_number != "unknown" and str(page_number).isdigit():
                quoted_text = doc.page_content
                page_info.append((int(page_number), quoted_text))

        if page_info:
            sorted_pages = sorted(list(set(page_num for page_num, _ in page_info)))
            page_numbers_str = ", ".join(map(str, sorted_pages))
            st.markdown("---")
            st.markdown(
                f"**{'المصدر' if interface_language == 'العربية' else 'Source'}:** " +
                f"{'صفحة رقم' if interface_language == 'العربية' else 'Page'} {page_numbers_str}"
            )

            cols = st.columns(2)
            for idx, (page_num, quoted_text) in enumerate(page_info):
                col_idx = idx % 2
                with cols[col_idx]:
                    screenshots = pdf_searcher.capture_screenshots(pdf_path, [(page_num, quoted_text)])
                    if screenshots:
                        st.image(
                            screenshots[0],
                            use_container_width=True,
                            width=300
                        )
                        st.markdown(
                            f"<div style='text-align: center;'>"
                            f"<p><strong>{'صفحة' if interface_language == 'العربية' else 'Page'} {page_num}</strong></p>"
                            f"<p><em>{quoted_text}</em></p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

def display_response_with_references(response, assistant_response):
    """عرض الرد مع المراجع"""
    if not any(phrase in assistant_response for phrase in negative_phrases):
        message = {
            "role": "assistant",
            "content": assistant_response,
            "references": {"context": response["context"]}
        }
        display_chat_message(message, with_refs=True)
    else:
        display_chat_message({
            "role": "assistant",
            "content": assistant_response
        })

# الكود الرئيسي
if __name__ == "__main__":
    st.set_page_config(page_title="BGC Assistant", page_icon="🤖", layout="wide")
    
    # تهيئة المتغيرات في session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            output_key="answer",
            input_key="input"
        )
    
    # تكوين الشريط الجانبي
    with st.sidebar:
        interface_language = st.selectbox(
            "اختر لغة الواجهة / Select Interface Language",
            ["العربية", "English"],
            index=0
        )
        
        # حقول API
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key here"
        )
        
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key here"
        )
        
        # حقل مسار PDF
        pdf_path = st.text_input(
            "PDF Path",
            value="",
            help="Enter the path to your PDF file"
        )
    
    # تهيئة المكونات إذا تم توفير المفاتيح
    if groq_api_key and google_api_key and pdf_path:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # تهيئة النماذج والأدوات
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
        pdf_searcher = PDFSearchAndDisplay()
        
        # تحميل التضمينات
        if "vectors" not in st.session_state:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.vectors = FAISS.load_local("faiss_index", embeddings)
            except Exception as e:
                st.error(f"خطأ في تحميل التضمينات: {str(e)}")
        
        # عرض سجل المحادثة
        for message in st.session_state.messages:
            if message["role"] == "assistant" and "references" in message:
                display_chat_message(message, with_refs=True)
            else:
                display_chat_message(message)
        
        # حقل الإدخال
        human_input = st.chat_input(
            "اكتب سؤالك هنا..." if interface_language == "العربية" else "Type your question here..."
        )
        
        # معالجة الإدخال
        if human_input:
            user_message = {"role": "user", "content": human_input}
            st.session_state.messages.append(user_message)
            display_chat_message(user_message)
            
            if "vectors" in st.session_state and st.session_state.vectors is not None:
                try:
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
                        
                        display_response_with_references(response, response["answer"])
                except Exception as e:
                    st.error(f"حدث خطأ: {str(e)}")
    else:
        st.warning("الرجاء إدخال مفاتيح API ومسار ملف PDF للبدء.")
