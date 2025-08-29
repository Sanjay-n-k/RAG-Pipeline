# core/vectorstore.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vectorstore(chunks, persist_dir: str = "data/chroma_db"):
    """
    Create a new Chroma vectorstore from a list of document chunks.
    Overwrites any existing collection at persist_dir.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_collection",
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb


def load_vectorstore(persist_dir: str = "data/chroma_db"):
    """
    Load an existing Chroma vectorstore (without adding new documents).
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma(
        embedding_function=embeddings,
        collection_name="rag_collection",
        persist_directory=persist_dir
    )
    return vectordb


def add_to_vectorstore(chunks, persist_dir: str = "data/chroma_db"):
    """
    Add new chunks to an existing ChromaDB vectorstore.
    If DB doesn't exist yet, it will create one.
    """
    vectordb = load_vectorstore(persist_dir)
    vectordb.add_documents(chunks)
    vectordb.persist()
    return vectordb
