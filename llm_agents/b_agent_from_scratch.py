import json
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

def add(x, y):
    """Return the sum of x and y"""
    return y - x

def product(x, y):
    """Return the product of x and y"""
    return x * y

def my_own_agent(prompt):
    assistant_prompt = """
    Given the prompt, get the function to call and the variables to use.
    
    These are the functions you have at disposal and their description.
    
    - add: return the sum of x and y
    - product: return the product of x and y
    
    Your answer should return exactly a dictionnay with these keys with the right function to call. Example with the add function:
    {"function": "add","variables": [1, 2]}
    
    Example with the product function:
    {"function": "product","variables": [1, 2]}
    
    If no function is adapted to the question, simply return {"function": "","variables": []}
    
    Here is the prompt:\n\n
    """
    
    func_dict = {
        "add": add,
        "product": product,
    }
    
    response = llm.invoke([HumanMessage(content=assistant_prompt + prompt)])
    
    try:
        print(response.content)
        data = json.loads(response.content)
        print(data)
        if func_dict.get(data["function"]):
            return func_dict[data["function"]](*data["variables"])
    except Exception as e:
        return "Error: " + e

if __name__ == "__main__":
    print(my_own_agent("Hi Sir, what is the result of 1 + 2"))
    print(my_own_agent("Hi Sir, what is the result of 1 * 2"))