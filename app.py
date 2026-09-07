import streamlit as st
from document_parser import parse_file
from chunking import split_documents
from embeddings import get_embeddings_model
from vector_store import create_vector_store
from rag_chain import build_rag_chain

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Q&A Assistant")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document upload & processing
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Parsing & indexing documents..."):
            try:
                all_docs = []
                for uploaded_file in uploaded_files:
                    docs = parse_file(uploaded_file)
                    all_docs.extend(docs)
                
                chunks = split_documents(all_docs)
                embeddings = get_embeddings_model()
                vector_store = create_vector_store(chunks, embeddings)
                
                st.session_state.rag_chain = build_rag_chain(vector_store)
                st.success(f"Successfully indexed {len(chunks)} text chunks!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

# Chat Interface
st.subheader("💬 Chat with your Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if not st.session_state.rag_chain:
        with st.chat_message("assistant"):
            st.warning("Please upload and process at least one document first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching context & generating response..."):
                try:
                    response = st.session_state.rag_chain.invoke(user_input)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Groq API Error: {str(e)}")
