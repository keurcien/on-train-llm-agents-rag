from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma

from langfuse.callback import CallbackHandler


from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

langfuse_handler = CallbackHandler()

loader = PyPDFLoader("data/cards.pdf")

documents = loader.load()
documents = [document for document in documents if document.metadata["page"] not in [0, 25, 32]]


class Card(BaseModel):
    name: str = Field(alias="Card name")
    team: str = Field(alias="Team")
    power: str = Field(alias="Character special power")
    role: str = Field(alias="Character role description")

prompt = ChatPromptTemplate.from_template("""
    Parse the following PDF to extract the name, team, and role of each card.
    Here's card information:
                                          
    # Card
    {card_content}
    """
)

chain = prompt | ChatOpenAI().with_structured_output(Card)

for document in documents:
    card = chain.invoke(document.page_content)
    document.metadata["name"] = card.name
    document.metadata["team"] = card.team

################


metadata_field_info = [
    AttributeInfo(
        name="name",
        description="The name of the character",
        type="string"
    ),
    AttributeInfo(
        name="team",
        description="The team of the character, whether it belongs to the Loup-Garous team or the Villageois team",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The page number",
        type="integer"
    )
]

document_content_description = "Brief description of the card character" 

llm = ChatOpenAI()

embeddings = OpenAIEmbeddings()

index = Chroma.from_documents(documents, embeddings)

retriever = SelfQueryRetriever.from_llm(
    llm,
    index,
    document_content_description,
    metadata_field_info,
    verbose=True
)

print(retriever.invoke("Who is the character from page 4", {"callbacks":[langfuse_handler]}))
print(retriever.invoke("Who is in the team Neutre and has to charm?", {"callbacks":[langfuse_handler]}))