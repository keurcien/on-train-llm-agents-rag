from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = OpenAIEmbeddings()

user_input = "Where is my cat"

docs = ["Your cat is in the kitchen.", "Your cat is in the kitchen. My parents are cooking dinner.", "Your cat is in the kitchen. My parents are cooking dinner. Tom is playing in the living room. The newspaper is on the table.", "My dog has eaten my homework", "Griezmann retired from the French team in september 2024", "There has been a lot of interest in the new iPhone 16"]

user_input_vector = [embeddings.embed_query(user_input)]

for doc in docs:
    doc_vector = [embeddings.embed_query(doc)]
    distance = cosine_similarity(user_input_vector, doc_vector)
    print(f"####")
    print(f"User input: {user_input}")
    print(f"Doc: {doc}")
    print(f"Similarity: {distance[0][0]}\n\n")
