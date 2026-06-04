import os
import logging
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ==================================================
# CONFIG
# ==================================================

FAISS_INDEX_PATH = "faiss_index"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_K = 5

MAX_K = 20

SIMILARITY_THRESHOLD = 1.2


# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# EMBEDDINGS
# ==================================================

def get_embeddings():

    try:

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    except Exception as e:

        logger.error(
            f"Embedding model failed: {e}"
        )

        raise


# ==================================================
# LOAD VECTORSTORE
# ==================================================

def load_vectorstore():

    if not os.path.exists(
        FAISS_INDEX_PATH
    ):

        raise FileNotFoundError(
            f"FAISS index not found at "
            f"{FAISS_INDEX_PATH}"
        )

    try:

        embeddings = get_embeddings()

        vectorstore = FAISS.load_local(

            FAISS_INDEX_PATH,

            embeddings,

            allow_dangerous_deserialization=True
        )

        logger.info(
            "FAISS loaded successfully."
        )

        return vectorstore

    except Exception as e:

        logger.error(
            f"Failed loading FAISS: {e}"
        )

        raise


# ==================================================
# SINGLETON VECTORSTORE
# ==================================================

try:

    VECTORSTORE = load_vectorstore()

except Exception as e:

    logger.error(e)

    VECTORSTORE = None


# ==================================================
# QUERY CLEANING
# ==================================================

def clean_query(
    query: str
) -> str:

    if not query:

        return ""

    query = query.strip()

    query = " ".join(
        query.split()
    )

    return query


# ==================================================
# DUPLICATE REMOVAL
# ==================================================

def remove_duplicates(
    docs: List[Document]
) -> List[Document]:

    unique_docs = []

    seen = set()

    for doc in docs:

        text = doc.page_content.strip()

        if text in seen:
            continue

        seen.add(text)

        unique_docs.append(doc)

    return unique_docs


# ==================================================
# RETRIEVAL
# ==================================================

def retrieve_documents(
    query: str,
    k: int = DEFAULT_K
) -> List[Document]:

    if VECTORSTORE is None:

        logger.error(
            "Vectorstore unavailable."
        )

        return []

    query = clean_query(query)

    if not query:

        logger.warning(
            "Empty query received."
        )

        return []

    k = min(
        max(1, k),
        MAX_K
    )

    try:

        docs = VECTORSTORE.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=20
        )

        filtered_docs = [doc for doc, score in results]

        filtered_docs = (
            remove_duplicates(
                filtered_docs
            )
        )

        logger.info(
            f"Retrieved "
            f"{len(filtered_docs)} docs."
        )

        return filtered_docs

    except Exception as e:

        logger.error(
            f"Retrieval failed: {e}"
        )

        return []


# ==================================================
# CONTEXT CREATION
# ==================================================

def get_context(
    query: str,
    k: int = DEFAULT_K
) -> str:

    docs = retrieve_documents(
        query,
        k
    )

    if not docs:

        return ""

    context_parts = []

    for doc in docs:

        text = doc.page_content.strip()

        if text:

            context_parts.append(
                text
            )

    return "\n\n".join(
        context_parts
    )


# ==================================================
# RESPONSE FOR STREAMLIT
# ==================================================

def get_retrieval_response(
    query: str,
    k: int = DEFAULT_K
) -> Dict:

    docs = retrieve_documents(
        query,
        k
    )

    if not docs:

        return {

            "success": False,

            "context": "",

            "sources": [],

            "documents": []
        }

    context = "\n\n".join(

        doc.page_content

        for doc in docs
    )

    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        sources.append(source)

    return {

        "success": True,

        "context": context,

        "sources": list(
            set(sources)
        ),

        "documents": docs
    }