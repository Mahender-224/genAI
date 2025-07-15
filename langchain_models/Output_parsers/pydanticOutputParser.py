from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field

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

class Person(BaseModel):
    name: str = Field(description = 'Name of the person')
    age: int = Field(gt = 18, description = 'Age of the person')
    city: str = Field(description = 'Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = "Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables = ['place'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

final_result = chain.invoke({'place': 'sri lankan'})

print(final_result)