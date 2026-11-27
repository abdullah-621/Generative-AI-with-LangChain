from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# 1st prompt -> detailed report
template1 = PromptTemplate(
  template="Write a detailed report on {topic}",
  input_variables=['topic']
)

# 2st prompt -> summary
template2 = PromptTemplate(
  template="Write a 5 line summary on the following text. /n {text}",
  input_variables=['text']
)

# prompt1 = template1.invoke({'topic' : "balck hole"})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({'text' : result.content})

# result1 = model.invoke(prompt2)

# print(result1.content)

# todo : Replace the code using the parser and chain . ekhane r manually ektar output r ektar vitore add korte hbe na chain er maddhome emnitei add hoye jabe

parser = StrOutputParser() # AIMessage theke plain string extract kore

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "balck hole"})

print(result)
