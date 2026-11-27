
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

import os

load_dotenv()

os.environ["HF_HOME"] = "D:/huggingface_cache"

# Load local model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 200
    }
)

model = ChatHuggingFace(llm=llm)

# Define output schema
schema = [
    ResponseSchema(
        name="fact_1",
        description="First fact about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Second fact about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Third fact about the topic"
    )
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Create prompt
template = PromptTemplate(
    template="""
Give 3 facts about {topic}.

{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# Create chain
chain = template | model | parser

# Invoke chain
result = chain.invoke({
    "topic": "black hole"
})

print(result)
