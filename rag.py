import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain.chains import RetrievalQA
import os
import glob

st.set_page_config(page_title="RAG HR Assistant", layout="wide")
st.title("🤖 RAG HR Assistant – Chatbot for HR Policies")

# Ensure Gemini API key is set
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
if not GEMINI_API_KEY:
    st.error("Google Gemini API key not found. Please set the GOOGLE_API_KEY environment variable, add it to Streamlit secrets, or put it in a .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Load and cache FAISS index
@st.cache_resource
def load_or_create_vectorstore():
    pdf_files = glob.glob(os.path.join("docs", "*.pdf"))
    if not pdf_files:
        st.error("No PDF files found in the docs directory.")
        return None
    try:
        st.info("🔄 Loading existing FAISS index...")
        return FAISS.load_local("local_faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    except:
        st.warning("⚙️ Creating FAISS index from docs folder...")
        all_documents = []
        for file_path in pdf_files:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(all_documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(docs, embedding=embeddings)
        vectordb.save_local("local_faiss_index")
        return vectordb

# Indexing from docs folder
vectordb = load_or_create_vectorstore()
if vectordb:
    retriever = vectordb.as_retriever(search_type="similarity", k=4)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
        retriever=retriever,
        return_source_documents=True
    )

    # User query input
    query = st.text_input("💬 Ask a question about HR policies:")
    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            st.success("✅ Answer:")
            st.write(result["result"])

            with st.expander("📚 Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}")
else:
    st.info("📂 Please add PDF documents to the docs directory to get started.")
