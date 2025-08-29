# core/rag_chain.py
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from .vectorstore import get_vectorstore

load_dotenv()  # load GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

def get_rag_chain():
    """Build RAG pipeline with retriever + Gemini."""
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are a helpful assistant. Use the following context to answer.
    Cite sources using [1], [2], etc.

    Context:
    {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain
