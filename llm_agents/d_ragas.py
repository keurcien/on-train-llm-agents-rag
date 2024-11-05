from dotenv import load_dotenv
load_dotenv()

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample, EvaluationDataset

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent

from duckduckgo_search import DDGS

load_dotenv()

ddg = DDGS()

llm = ChatOpenAI()

@tool
def search(query):
    """Search the web for a query"""
    search_result = ddg.text(query, max_results=1)
    return search_result


system_prompt = ChatPromptTemplate.from_template("""You are a football expert. You are asked questions about football. You must answer the questions as accurately as possible.
    If you don't know, you can use the search tool to find the answer. Only use results from the search tool to answer the question. Do not make up information.
""")

agent = create_react_agent(llm, [search])

samples = [
    {
        "user_input": "Who won the UCL game in 2024 between PSG and PSV Eindhoven?",
        "response": "It was a draw",
    },
    {
        "user_input": "When did Antoine Griezmann retire from the French National Team?",
        "response": "Griezmann retires from the French team in september 2024",
    },    
    {
        "user_input": "Who won the Ballon d'Or in 2024?",
        "response": "Rodri",
    },    
    {
        "user_input": "Where is Olivier Giroud playing in 2024?",
        "response": "Los Angeles FC",
    }
]

ragas_samples = []
for sample in samples:
    user_input = sample["user_input"]
    output = agent.invoke({"messages": [sample["user_input"]]})
    tool_message = [m for m in output["messages"] if isinstance(m, ToolMessage)][-1]
    answer = output["messages"][-1].content
    ragas_samples.append(SingleTurnSample(
        user_input=user_input,
        response=answer,
        reference=sample["response"],
        retrieved_contexts=[tool_message.content if tool_message else ""]
    ))

dataset = EvaluationDataset(samples=ragas_samples)
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=dataset, metrics=metrics)

df = results.to_pandas()
df.to_csv("outputs/football_expert.csv", index=False)