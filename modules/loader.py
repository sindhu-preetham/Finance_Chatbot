from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://www.moneycontrol.com/news/business/"
]

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

print(data)