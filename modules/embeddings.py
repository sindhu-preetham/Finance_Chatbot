import os
import logging
import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

FAISS_INDEX_PATH = "faiss_index"

EMBEDDING_MODEL = "text-embedding-3-small"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_documents(
    documents: List[Document]
):

    if not documents:

        raise ValueError(
            "No documents supplied for embedding."
        )

    valid_docs = []

    for doc in documents:

        if not doc.page_content:
            continue

        if len(doc.page_content.strip()) < 10:
            continue

        valid_docs.append(doc)

    if not valid_docs:

        raise ValueError(
            "All documents were empty."
        )

    return valid_docs


# --------------------------------------------------
# REMOVE DUPLICATE CHUNKS
# --------------------------------------------------

def remove_duplicate_documents(
    documents: List[Document]
):

    unique_docs = []

    seen = set()

    for doc in documents:

        content_hash = hashlib.md5(
            doc.page_content.encode("utf-8")
        ).hexdigest()

        if content_hash in seen:
            continue

        seen.add(content_hash)

        unique_docs.append(doc)

    logger.info(
        f"Unique documents: {len(unique_docs)}"
    )

    return unique_docs


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# CREATE NEW VECTOR STORE
# --------------------------------------------------

def create_vectorstore(
    documents: List[Document]
):

    documents = validate_documents(
        documents
    )

    documents = remove_duplicate_documents(
        documents
    )

    embeddings = get_embedding_model()

    logger.info(
        "Creating FAISS index..."
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    logger.info(
        f"Indexed {len(documents)} chunks."
    )

    return vectorstore


# --------------------------------------------------
# SAVE INDEX
# --------------------------------------------------

def save_vectorstore(
    vectorstore,
    path=FAISS_INDEX_PATH
):

    logger.info(
        f"Saving FAISS index to {path}"
    )

    vectorstore.save_local(path)

    logger.info(
        "FAISS index saved."
    )


# --------------------------------------------------
# LOAD INDEX
# --------------------------------------------------

def load_vectorstore(
    path=FAISS_INDEX_PATH
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{path} does not exist."
        )

    embeddings = get_embedding_model()

    logger.info(
        f"Loading index from {path}"
    )

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logger.info(
        "FAISS index loaded."
    )

    return vectorstore


# --------------------------------------------------
# APPEND NEW DOCUMENTS
# --------------------------------------------------

def add_documents_to_index(
    vectorstore,
    documents: List[Document]
):

    documents = validate_documents(
        documents
    )

    documents = remove_duplicate_documents(
        documents
    )

    vectorstore.add_documents(
        documents
    )

    logger.info(
        f"Added {len(documents)} chunks."
    )

    return vectorstore


# --------------------------------------------------
# INDEX STATS
# --------------------------------------------------

def print_index_stats(
    documents: List[Document]
):

    total_chars = sum(
        len(doc.page_content)
        for doc in documents
    )

    avg_chars = (
        total_chars // len(documents)
    )

    logger.info(
        f"Chunks: {len(documents)}"
    )

    logger.info(
        f"Average Length: {avg_chars}"
    )

    logger.info(
        f"Total Characters: {total_chars}"
    )


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    from loader import load_articles
    from chunking import create_chunks

    urls = [

        "https://www.moneycontrol.com/news/business/",

        "https://www.livemint.com/"
    ]

    docs = load_articles(urls)

    chunks = create_chunks(docs)

    print_index_stats(
        chunks
    )

    vectorstore = create_vectorstore(
        chunks
    )

    save_vectorstore(
        vectorstore
    )

    print(
        "\nFAISS Index Created Successfully\n"
    )