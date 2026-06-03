# 📈 Finance News Chatbot

A Retrieval-Augmented Generation (RAG) based Finance News Chatbot that allows users to analyze financial news articles and ask natural language questions about them.

The project automatically extracts content from finance-related news URLs, processes the articles, creates searchable text chunks, and enables question answering using Large Language Models (LLMs).

---

## 🚀 Project Overview

Financial news is often spread across multiple websites and articles. Reading and extracting insights manually can be time-consuming.

This chatbot helps users:

* Load financial news articles directly from URLs
* Extract clean article content
* Process and chunk large articles
* Build a searchable knowledge base
* Ask questions in natural language
* Receive contextual answers based on article content

Examples:

* "What is the analyst target price for Tata Motors?"
* "What are the key reasons for the stock recommendation?"
* "Summarize the article in 5 points."
* "What risks are mentioned in the article?"

---

## 🏗️ System Architecture

```text
User URLs
    ↓
Article Loader
    ↓
Content Extraction
    ↓
Text Cleaning
    ↓
Chunking Pipeline
    ↓
Embeddings
    ↓
FAISS Vector Database
    ↓
Retriever
    ↓
LLM
    ↓
Answer Generation
```

---

## 📂 Project Structure

```text
Finance_Chatbot/
│
├── app.py
├── requirements.txt
│
├── modules/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py          (Planned)
│   ├── vector_store.py        (Planned)
│   ├── retriever.py           (Planned)
│   └── rag_chain.py           (Planned)
│
└── README.md
```

---

## ✅ Completed Features

### Article Loader

* URL validation
* Duplicate URL removal
* Retry mechanism
* Custom browser headers
* Content extraction using Trafilatura
* BeautifulSoup fallback extraction

### Chunking Pipeline

* Text cleaning
* Paragraph-based splitting
* Recursive chunking
* Configurable chunk size and overlap
* Duplicate chunk removal
* Metadata generation

### Streamlit Interface

* User-friendly URL input
* Article processing workflow
* Question submission interface
* Chat history display

---

## 🚧 Work In Progress

* Embedding generation
* FAISS vector database integration
* Semantic retrieval
* RAG pipeline
* Source attribution
* Answer citations
* Deployment

---

## 🛠️ Technologies Used

### Frontend

* Streamlit

### Backend

* Python

### NLP & AI

* LangChain
* Trafilatura
* BeautifulSoup
* FAISS (Planned)
* OpenAI / HuggingFace Embeddings (Planned)

### Data Processing

* Recursive Character Text Splitter
* Document Processing Pipeline

---

## 🎯 Future Enhancements

* Multi-article comparison
* Financial sentiment analysis
* Stock-specific question answering
* Article summarization
* Portfolio-related insights
* Real-time news integration
* Chat history persistence

---

## 👥 Contributors

### Pooja H M

* RAG Architecture
* Embeddings & Retrieval Pipeline
* Vector Database Integration
* LLM Integration

### Sindhu Preetham

* Article Loader Development
* Chunking Pipeline
* Streamlit User Interface
* Data Processing Workflow

---

## 📌 Current Status

The project currently supports article extraction, preprocessing, and chunk generation. The next phase focuses on implementing the Retrieval-Augmented Generation (RAG) pipeline using embeddings, vector search, and large language models.
