from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://www.moneycontrol.com/news/business/"
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(data)

print("Number of chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0].page_content)