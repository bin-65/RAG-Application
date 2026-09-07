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
    
    # Retrieve & clean Groq API Key
    raw_key = st.secrets.get("GROQ_API_KEY", "")
    api_key = str(raw_key).strip().strip('"').strip("'")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in Streamlit secrets.")

    # Model configured to openai/gpt-oss-120b
    llm = ChatGroq(
        temperature=0.2,
        model_name="openai/gpt-oss-120b",
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
