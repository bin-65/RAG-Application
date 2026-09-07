from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embeddings_model):
    return FAISS.from_documents(chunks, embeddings_model)
