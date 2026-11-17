from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()

# Load Hugging Face Model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 500
    }
)

Model = ChatHuggingFace(llm=llm)

message = [
  SystemMessage(content="You are a helpful assistant."),
  HumanMessage(content="Tell me about langchain.")
]

result = Model.invoke(message)

message.append(AIMessage(content=result.content))

print(message)