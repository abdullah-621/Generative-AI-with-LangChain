
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

# os.environ["HF_HOME"] = "D:/huggingface_cache"

# # Load Hugging Face Model
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=150
#     )
# )

# model = ChatHuggingFace(llm=llm)

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.5
)


# Define Pydantic Schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, lt=60, description="Age of the person")
    city: str = Field(description="City of the person")


# Create Output Parser
parser = PydanticOutputParser(
    pydantic_object=Person
)


# Create Prompt Template
template = PromptTemplate(
    template="""
You are a data generator.

Generate ONE fictional person from {place}.

Return ONLY a JSON object.

Example:
{{
    "name": "Rahim",
    "age": 25,
    "city": "Dhaka"
}}

Do not return JSON schema.
Do not explain anything.
Do not include properties or required.

{format_instruction}
""",
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# Create LangChain Chain
chain = template | model | parser


# Invoke Chain
final_result = chain.invoke({
    "place": "Bangladesh"
})

print(final_result)

# Access Individual Fields
print(final_result.name)
print(final_result.age)
print(final_result.city)
