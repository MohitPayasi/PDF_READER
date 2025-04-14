# PDF_READER
This project is a **Research Paper Question & Answer** system built using **LangChain**, **Streamlit**, and **Retrieval-Augmented Generation (RAG)** techniques. The system allows users to upload research papers in PDF format, processes them to create embeddings, and provides accurate answers to user queries based on the content of the uploaded documents.

## Features
- **PDF Document Upload**: Users can upload multiple PDF files containing research papers.
- **Document Preprocessing**: The uploaded PDF documents are processed and split into smaller chunks for better retrieval and understanding.
- **Vector Embeddings**: The system creates vector embeddings using pre-trained models like **Ollama** (or other customizable models) for document retrieval.
- **Question Answering**: Users can ask questions related to the content of the uploaded research papers, and the system will return relevant answers based on the documents.
- **Retrieval-Augmented Generation (RAG)**: Uses a combination of document retrieval and a language model (such as **Groq** or **Gemma**) to generate accurate answers based on the provided context.

## Key Components
- **Streamlit**: For creating the web interface to upload files and display answers.
- **LangChain**: For handling document loading, vector embedding, and chaining the retrieval-augmented generation model.
- **FAISS**: For efficient similarity search to retrieve relevant document chunks from the vector store.
- **Ollama Embeddings**: For creating embeddings (can be replaced with another embedding model if desired).
- **Groq API**: Used for generating responses based on the context from the documents.
