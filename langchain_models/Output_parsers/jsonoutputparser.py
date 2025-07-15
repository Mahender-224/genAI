from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Step 1: Authenticated client
client = InferenceClient(
    model="google/gemma-2-2b-it", 
    token="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"
)

# Step 2: Pass both `client` and `repo_id`
llm = HuggingFaceEndpoint(
    client=client,
    repo_id="google/gemma-2-2b-it",  # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    timeout=300
)

# Step 3: Create chat model and invoke
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

# above commented lines can be written as 
chain = template | model | parser

result = chain.invoke({})

print(result)