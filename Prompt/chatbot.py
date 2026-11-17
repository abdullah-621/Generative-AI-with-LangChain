
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load Hugging Face Model
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 500,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }
)

# Convert LLM into Chat Model
model = ChatHuggingFace(llm=llm)

chat_history = [
  SystemMessage(content="You are a helper assistant")
]

# Chat Loop
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower().strip() == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)
