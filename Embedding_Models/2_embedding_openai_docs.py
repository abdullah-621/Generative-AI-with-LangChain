from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)

documents = [
  'Dhaka is the capital of bangladesh',
  'Kolkata is the capital of west bengle',
  'Paris is the capital of France'
]

result = embedding.embed_documents(documents)

print(str(result))