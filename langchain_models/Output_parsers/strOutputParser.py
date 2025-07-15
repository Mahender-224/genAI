# here we are fetching the model from Huggingface and caling api
from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({'topic':'black hole'})

result1 = chat_model.invoke(prompt1)
print(result1.content)

prompt2 = template2.invoke({'text': result1.content})

result2 = chat_model.invoke(prompt2)

print(result2.content)