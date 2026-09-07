import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def parse_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    try:
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "docx":
            loader = Docx2txtLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        documents = loader.load()
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return documents
