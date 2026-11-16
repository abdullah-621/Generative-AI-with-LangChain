from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError("Hugging Face API token not found!")

llm = HuggingFaceEndpoint(
    repo_id="swiss-ai/Apertus-70B-Instruct-2509",
    task="text-generation",
    provider="publicai",
    huggingfacehub_api_token=token,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Bangladesh?")

print(result.content)
