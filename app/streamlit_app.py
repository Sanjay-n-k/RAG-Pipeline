import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from core.ingestion import load_and_split_document
from core.vectorstore import create_vectorstore, load_vectorstore, add_to_vectorstore

# Load environment
load_dotenv()

st.set_page_config(page_title="📚 RAG with Gemini", layout="centered")

st.title("📚 RAG System with Gemini 2.5 Flash")
st.write("Upload a PDF, create a knowledge base, and ask questions with **Gemini-powered answers**.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ Uploaded {uploaded_file.name}")

    # Process document
    if st.button("🔄 Ingest & Create Vectorstore"):
        with st.spinner("Processing document..."):
            chunks = load_and_split_document(file_path)
            vectordb = create_vectorstore(chunks)
        st.success("Vectorstore created successfully!")

    if st.button("➕ Add to Existing Vectorstore"):
        with st.spinner("Adding document..."):
            chunks = load_and_split_document(file_path)
            vectordb = add_to_vectorstore(chunks)
        st.success("Added document to existing vectorstore!")

# Query
st.subheader("Ask a Question")
query = st.text_input("Type your question about the uploaded documents:")

if query:
    with st.spinner("Retrieving answer..."):
        vectordb = load_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Get docs
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        # Build prompt for Gemini
        prompt = f"""
        You are an assistant answering questions based on a document.
        Use the following context to answer:

        {context}

        Question: {query}

        Give a concise answer. Include numbered citations like [1], [2], [3] matching the chunks.
        """

        # Call Gemini via LangChain
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )
        response = llm.invoke(prompt)

        # Display
        st.markdown("### 💡 Gemini Answer")
        st.write(response.content if hasattr(response, "content") else str(response))

        st.markdown("### 📚 Sources")
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**[{i}]** {doc.metadata}")
