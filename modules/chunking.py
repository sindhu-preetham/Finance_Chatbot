import hashlib
import logging
from urllib.parse import urlparse
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------------------
# CONFIG
# ----------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ----------------------------------------
# URL VALIDATION
# ----------------------------------------

def validate_url(url: str) -> bool:

    try:

        parsed = urlparse(url)

        return all([
            parsed.scheme in ["http", "https"],
            parsed.netloc
        ])

    except Exception:

        return False


# ----------------------------------------
# CLEAN TEXT
# ----------------------------------------

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\r", "")

    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    return "\n".join(lines)


# ----------------------------------------
# PARAGRAPH SPLITTING
# ----------------------------------------

def paragraph_split(
    document: Document
) -> List[Document]:

    paragraphs = document.page_content.split("\n")

    paragraph_docs = []

    for idx, paragraph in enumerate(paragraphs):

        paragraph = paragraph.strip()

        if len(paragraph) < 50:
            continue

        paragraph_docs.append(

            Document(

                page_content=paragraph,

                metadata={
                    **document.metadata,
                    "paragraph_id": idx
                }
            )
        )

    return paragraph_docs


# ----------------------------------------
# RECURSIVE SPLITTING
# ----------------------------------------

def recursive_split(
    docs: List[Document]
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_documents(
        docs
    )


# ----------------------------------------
# REMOVE DUPLICATES
# ----------------------------------------

def remove_duplicates(
    chunks: List[Document]
) -> List[Document]:

    unique = []

    seen = set()

    for chunk in chunks:

        hash_value = hashlib.md5(

            chunk.page_content.encode()

        ).hexdigest()

        if hash_value not in seen:

            seen.add(hash_value)

            unique.append(chunk)

    return unique


# ----------------------------------------
# MAIN CHUNKING PIPELINE
# ----------------------------------------

def create_chunks(
    documents: List[Document]
) -> List[Document]:

    if not documents:

        raise ValueError(
            "No documents received."
        )

    all_paragraphs = []

    for doc in documents:

        cleaned_text = clean_text(
            doc.page_content
        )

        if len(cleaned_text) < 100:

            logger.warning(
                "Skipping small document."
            )

            continue

        cleaned_doc = Document(

            page_content=cleaned_text,

            metadata=doc.metadata
        )

        paragraphs = paragraph_split(
            cleaned_doc
        )

        all_paragraphs.extend(
            paragraphs
        )

    logger.info(
        f"Paragraphs created: {len(all_paragraphs)}"
    )

    chunks = recursive_split(
        all_paragraphs
    )

    chunks = remove_duplicates(
        chunks
    )

    for idx, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = idx

        chunk.metadata["chunk_length"] = len(
            chunk.page_content
        )

    logger.info(
        f"Final chunks: {len(chunks)}"
    )

    return chunks
if __name__ == "__main__":

    from loader import load_articles

    urls = [
        "https://www.moneycontrol.com/news/business/"
    ]

    docs = load_articles(urls)

    chunks = create_chunks(docs)

    print("\nTotal Chunks:")
    print(len(chunks))

    print("\nFirst Chunk:")
    print(chunks[0].page_content)

    print("\nMetadata:")
    print(chunks[0].metadata)