from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Authenticated client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta", 
    token="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"
)

# Step 2: Pass both `client` and `repo_id`
llm = HuggingFaceEndpoint(
    client=client,
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    timeout=300
)

# Step 3: Create chat model and invoke
chat_model = ChatHuggingFace(llm=llm)

# prompt 1=> detailed report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic} and describe about it more simpler terms',
    input_variables = ['topic'],
    timeout=60
)

# prompt 2: summary
template2 = PromptTemplate(
    template = 'Write a 5 line summary on the following text. \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result.content)