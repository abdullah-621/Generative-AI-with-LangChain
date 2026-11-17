from langchain_core.prompts import PromptTemplate

# Prompt Template
template = PromptTemplate(
    template="""
You are an expert academic research assistant.

Analyze the following research paper:

Research Paper: {paper_input}

Explanation Style: {style_input}

Explanation Length: {length_input}

Provide a structured research analysis covering:

1. Research Objective
2. Problem Statement
3. Proposed Methodology
4. Dataset Used
5. Key Findings and Results
6. Research Contributions
7. Limitations
8. Potential Future Research Directions

Instructions:
- Use clear academic language.
- Follow the selected explanation style.
- Follow the selected explanation length.
- Avoid making up information.
- If information is unavailable, explicitly mention it.
""",
    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ]
)

template.save('template.json')