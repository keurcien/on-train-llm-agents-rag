from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_amount=0.7)

# Test du splitter
chunks = text_splitter.split_text(
    "Your cat is in the kitchen. The cat is eating there and is seeking for attention from you. My parents are cooking dinner. Tom is playing in the living room. The newspaper is on the table. My dog has eaten my homework. Griezmann retired from the French team in september 2024"    
)

print("\n\n".join(chunks))

# Fin du test


loader = TextLoader("data/cgv.md")
documents = loader.load()

chunks = text_splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    print(f"### chunk {i} ###")
    print(chunk.page_content)
    print("\n\n")

### Utilisation avec le retriever

from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import SemanticSimilarity
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample, EvaluationDataset

vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieval_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever)


samples = [
    {
        "user_input": "Que disent les conditions de parrainage ?",
        "response": "Les conditions de parrainage stipulent que le parrain peut obtenir un coupon de réduction lors de l'expédition de la première Commande passée pour un montant minimum de trente (30) euros pour chaque nouveau filleul parrainé. Le filleul parrainé ne doit pas déjà avoir acheté un Produit via l’Application. Un Parrain peut parrainer cinq (5) filleuls par an et pour chaque parrainage validé, le Parrain et le filleul recevront chacun un coupon d'une valeur de dix (10) euros.",
    }, 
    {
        "user_input": "Quand un produit est-il considéré comme conforme ?",
        "response": "Un produit est jugé conforme s'il correspond à la description donnée par le vendeur et possède les qualités que celui-ci a présentées à l'acheteur sous forme d'échantillon ou de modèle. Ou bien s'il présente les qualités qu'un acheteur peut légitimement attendre eu égard aux déclarations publiques faites par le vendeur, par le producteur ou par son représentant, notamment dans la publicité ou l'étiquetage.",
    },
    {
        "user_input": "Pour quelle raison le site utilise des cookies ?",
        "response": "Pour permettre aux Internautes de bénéficier des services proposés par le Site ou l’Application tels que sa consultation, l’optimisation de son utilisation ou sa personnalisation en fonction de l’Internaute, le Site ou l’Application utilise des Cookies..",
    }
]

ragas_samples = []
for sample in samples:
    user_input = sample["user_input"]
    output = retrieval_chain.invoke(sample["user_input"])
    answer = output["result"]
    ragas_samples.append(SingleTurnSample(
        user_input=user_input,
        response=answer,
        reference=sample["response"],
    ))

dataset = EvaluationDataset(samples=ragas_samples)
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=dataset, metrics=metrics)

df = results.to_pandas()
df.to_csv("outputs/cgv_expert.csv", index=False)