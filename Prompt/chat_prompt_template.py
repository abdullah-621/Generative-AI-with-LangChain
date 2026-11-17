from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from transformers import AutoTokenizer



chat_template = ChatPromptTemplate([
  ('system', 'you are a helpful {domain} expert'),
  ('human', 'Explain in terms, what is {topic} in {domain}')
])

promt = chat_template.invoke({'domain' : 'cricket', 'topic':'LBW'})

# print(promt)


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

results = model.invoke(promt)

print(results.content)