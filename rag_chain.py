import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from prompts import SYSTEM_PROMPT

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vector_store):
    """
    Builds the LCEL RAG chain integrating FAISS retriever and Groq LLM.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Retrieve Groq API Key from Streamlit secrets
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets.")

    # Using active Groq model
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        api_key=api_key
    )
    
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
