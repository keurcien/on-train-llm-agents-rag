from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_template("Génère une recette de cuisine en français contenant cet ingrédient {ingrédient}")

class Recipe(BaseModel):
    title: str =  Field(description="Recipe title")
    ingredients: List[str] = Field(description="List of ingredients")
    instructions: List[str] = Field(description="List of instructions")
    
structured_llm = llm.with_structured_output(Recipe)

chain = prompt | structured_llm
chain.invoke("guanciale")