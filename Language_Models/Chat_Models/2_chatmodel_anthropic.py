from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-opus-4-6')

result = model.invoke("What is the capital of BD?")

print(result.content)