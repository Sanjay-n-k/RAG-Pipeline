# RAG-Pipeline

Full-stack RAG system for document Q&A with citations, powered by Gemini 2.5 Flash and evaluated with Ragas. Built with LangChain, FastAPI.

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline that allows users to:

- Upload PDF documents and create a knowledge base
- Query documents using natural language questions
- Get AI-powered answers with citations powered by Google's Gemini 2.5 Flash
- Use both command-line interface and web-based Streamlit interface

## 🏗️ Architecture

The system follows a modular architecture with the following components:

- **Document Ingestion**: PDF loading and text chunking using LangChain
- **Vector Store**: ChromaDB for storing document embeddings
- **Retrieval**: Semantic search using HuggingFace embeddings
- **Generation**: Google Gemini 2.5 Flash for answer generation
- **Interface**: Streamlit web application for user interaction

## 📁 Project Structure

```
RAG-Pipeline/
├── app/
│   └── streamlit_app.py       # Streamlit web interface
├── core/
│   ├── ingestion.py           # Document loading and splitting
│   ├── vectorstore.py         # ChromaDB operations
│   └── rag_chain.py          # RAG pipeline setup
├── data/                      # Document storage (gitignored)
├── main.py                    # Command-line interface
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF documents
- **Intelligent Text Chunking**: Recursive character text splitting for optimal retrieval
- **Semantic Search**: Vector-based similarity search using HuggingFace embeddings
- **Citation Support**: AI responses include numbered citations referencing source documents
- **Persistent Vector Store**: ChromaDB for storing and retrieving document embeddings
- **Web Interface**: User-friendly Streamlit application
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

## 🛠️ Major Tools Used

- **LangChain**: RAG pipeline framework and document processing
- **Google Gemini 2.5 Flash**: Large language model for answer generation
- **ChromaDB**: Vector database for document embeddings storage
- **Streamlit**: Web application framework for user interface
- **HuggingFace Transformers**: Text embedding models (all-MiniLM-L6-v2)
- **PyPDF**: PDF document processing and text extraction