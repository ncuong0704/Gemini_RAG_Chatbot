import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# tải biến môi trường
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    st.error('Không tìm thấy key')
    st.stop()

genai.configure(api_key=api_key)

# helper functions
def get_pdf_text(pdf_docs):
    text = ''
    try:
        for pdf in pdf_docs:
            # tạo file tạm thời để lưu nội dung từ các file và sau đó sẽ xoá
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file_path = tmp_file.name
            
            pdf_reader = PyPDFLoader(tmp_file_path)
            for page in pdf_reader.load_and_split():
                text += page.page_content
            
            os.unlink(tmp_file_path) # xoá file tạm thời
    except Exception as e:
        st.error(f'Lỗi đọc file PDF: {str(e)}')
    
    return text
        
def get_text_chunk(text):
    try:
        #chunk_overlap là nếu từ 1 đến 10000 chưa đủ nghĩa ở cuối thì nó sẽ có lố thêm 1000 để đầy đủ nghĩa
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f'Lỗi chia chunk: {str(e)}')
        return []

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Tài liệu đã phân tích xong, sẵn sàng để trả lời câu hỏi.")
    except Exception as e:
        st.error(f'Lỗi lưu vector database: {str(e)}')

def get_conversational_chain():
    prompt_template = """
    Trả lời câu hỏi một cách chi tiết nhất có thể dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không có trong ngữ cảnh được cung cấp, hãy nói, "Câu trả lời không có trong ngữ cảnh."
    Không cung cấp thông tin sai lệch.

    Ngữ cảnh: {context}
    Câu hỏi: {question}

    Answer:
    """
    try:
        #temperature là độ sáng tạo, để 0.3 để focus vào tài liệu để ko lấy thông tin bên ngoài
        # nếu cho chỉ số cao thì mình phải finetune nó để nó làm theo ý mình
        model = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3)
        #context là tài liệu mình đưa vào
        #question là câu hỏi của user
        prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
        #chain_type='stuff' lấy tất các chunks và question đưa vào mô hình
        chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
        return chain
    except Exception as e:
        st.error(f'Lỗi trong quá trình phân tích: {str(e)}')
        return None

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists('faiss_index'):
            st.error('Không tìm thấy FAISS index. Hãy tải file pdf lên trước')
            return
        #allow_dangerous_deserialization=False báo lỗi khi có mã độc, True thì bỏ qua
        new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        print("test 1")
        if not chain:
            return
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f'Lỗi xử lý câu hỏi: {str(e)}')
    
# set up trang chính streamlit
st.set_page_config(page_title='Chat PDF RAG')
st.title('Chatbot phân tích tài liệu PDF')

user_question = st.text_input('Bạn hãy hỏi sau khi tài liệu đã được phân tích')

if user_question:
    user_input(user_question)

with st.sidebar:
    st.title('Menu')
    pdf_docs = st.file_uploader('Tải tài liệu pdf của bạn lên', accept_multiple_files=True, type=['pdf'])
    
    if st.button('Phân tích tài liệu'):
        if not pdf_docs:
            st.error("Vui lòng tải tài liệu lên trước")
        with st.spinner('Đang xử lý...'):
            raw_text = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks = get_text_chunk(raw_text)
                if text_chunks:
                    get_vector_store(text_chunks)
                else:
                    st.error('Kiểm tra lại nội dung tài liệu pdf.')

