import warnings
warnings.filterwarnings('ignore')

import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain & RAG tools
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader


# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_key = os.getenv('GROQ_API_KEY')
os.environ['Hugging_face_token']=os.getenv('Hugging_face_token')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']='PDF Reader'


# Initialize LLM & embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')
# Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_key, model='llama-3.3-70b-versatile')

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the provided question based on the given context.
Please provide the most accurate response possible.

<context>
{context}
</context>

Question: {input}
""")

# Streamlit app UI
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("📄 Upload & Ask Questions on Your PDF (RAG + Groq + Ollama)")

# Upload PDF
uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

# Build vector store from uploaded files
if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)

        st.session_state.vectors = vectors
        st.success("✅ Vector database created from uploaded documents!")

# Query input
query = st.text_input("Enter your question about the uploaded documents")

# Run RAG chain
if query:
    if 'vectors' not in st.session_state:
        st.error("Please upload documents first.")
    else:
        with st.spinner("Retrieving answer..."):
            retriever = st.session_state.vectors.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retriever_chain.invoke({'input': query})
            elapsed = time.process_time() - start

            st.subheader("My Answer:")
            st.write(response['answer'])
            st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

        # Show source documents
