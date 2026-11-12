from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-4.1-mini')

results = llm.invoke("What is the capital of Bangladesh?")

print(results.content)