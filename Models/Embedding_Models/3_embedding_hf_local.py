from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]


query = "Dhaka is the capital of bangladesh"

# vector = embedding.embed_documents(documents)

vector2 = embedding.embed_query(query)
print(vector2)

# print(str(vector))