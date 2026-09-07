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
    
    # Secrets retrieval & clean-up
    raw_key = st.secrets.get("GROQ_API_KEY", "")
    api_key = str(raw_key).strip().strip('"').strip("'")
    
    if not api_key or api_key in ["your_groq_api_key_here", "gsk_..."]:
        raise ValueError("Valid GROQ_API_KEY missing from Streamlit secrets.")

    # Active supported model on Groq API
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )
    
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
