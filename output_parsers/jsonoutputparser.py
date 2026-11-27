from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace


load_dotenv()

import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
  model_id= 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  task='text-generation',
  pipeline_kwargs=dict(
    temperature = .5,
    max_new_tokens=100
  )
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
  template='Give me 5 facts about {topic} \n {format_instruction}',
  input_variables=['topic'],
  partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({"topic": "Black Hole"})

# results = model.invoke(prompt)

# print(results.content)

chain = template | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)

