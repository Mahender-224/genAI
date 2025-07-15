# here we are fetching the model from Huggingface and caling api
from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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
)

# Step 3: Create chat model and invoke
chat_model = ChatHuggingFace(llm=llm)
response = chat_model.invoke("what is capital of india and describe about it in short?")
print(response.content)
#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
