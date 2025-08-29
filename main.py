# main.py
from core.ingestion import load_and_split_document
from core.vectorstore import add_to_vectorstore
from core.rag_chain import get_rag_chain

def main():
    # Step 1: Load and embed a PDF
    file_path = "data/sample.pdf"   # <-- put any small PDF here
    chunks = load_and_split_document(file_path)
    add_to_vectorstore(chunks)

    # Step 2: Initialize RAG chain
    chain = get_rag_chain()

    # Step 3: Ask a test question
    query = "What is this document about?"
    response = chain(query)

    print("\n💡 Answer:")
    print(response['result'])

    print("\n📚 Sources:")
    for doc in response['source_documents']:
        print(doc.metadata)

if __name__ == "__main__":
    main()
