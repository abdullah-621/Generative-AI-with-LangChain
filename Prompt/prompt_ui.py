
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import PromptTemplate, load_prompt

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

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

# # Prompt Template
# template = PromptTemplate(
#     template="""
# You are an expert academic research assistant.

# Analyze the following research paper:

# Research Paper: {paper_input}

# Explanation Style: {style_input}

# Explanation Length: {length_input}

# Provide a structured research analysis covering:

# 1. Research Objective
# 2. Problem Statement
# 3. Proposed Methodology
# 4. Dataset Used
# 5. Key Findings and Results
# 6. Research Contributions
# 7. Limitations
# 8. Potential Future Research Directions

# Instructions:
# - Use clear academic language.
# - Follow the selected explanation style.
# - Follow the selected explanation length.
# - Avoid making up information.
# - If information is unavailable, explicitly mention it.
# """,
#     input_variables=[
#         "paper_input",
#         "style_input",
#         "length_input"
#     ]
# )

template = load_prompt('template.json')

# Generate Prompt
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

# Generate Response
if st.button("Summarize"):
    with st.spinner("Analyzing Research Paper..."):
        response = model.invoke(prompt)
        st.write(response.content)
