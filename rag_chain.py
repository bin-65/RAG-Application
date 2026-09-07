import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from prompts import SYSTEM_PROMPT

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Retrieve Groq API Key
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key or api_key.strip() in ["", "your_groq_api_key_here"]:
        raise ValueError("GROQ_API_KEY is missing or invalid in Streamlit secrets.")

    # Fixed model name to currently supported endpoint
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",  # Updated active Groq model ID
        groq_api_key=api_key.strip()
    )
    
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
