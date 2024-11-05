from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Génère un mot commencençant par la lettre {letter}")
parser = StrOutputParser()

chain = prompt | llm | parser

if __name__ == "__main__":
    output = chain.invoke("a")
    print(output)