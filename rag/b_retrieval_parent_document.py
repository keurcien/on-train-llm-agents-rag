from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langfuse.callback import CallbackHandler


from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

langfuse_handler = CallbackHandler()

llm = ChatOpenAI(temperature=0)

loaders = [
    PyPDFLoader("data/cgu.pdf"),
    PyPDFLoader("data/ciqual.pdf"),
]

documents = []
for loader in loaders:
    documents.extend(loader.load())
    
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings())

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

output = retrieval_chain.invoke("Quelle est la définition réglementaire des glucides ? Tu ajouteras la source de l'information dans la réponse", {"callbacks":[langfuse_handler]})
print(output)