from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/cgv.md")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separators=["#", "**"])

chunks = text_splitter.split_text(documents[0].page_content)

for i, chunk in enumerate(chunks):
    print(f"### chunk {i} ###")
    print(chunk)
