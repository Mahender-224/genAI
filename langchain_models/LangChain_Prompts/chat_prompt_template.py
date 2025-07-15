from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content = "You are helpful {domain} expert"),
#     HumanMessage(content= "Explain in a simple terms, what is {topic}")
# ])

# Now if we try to replace placeholder in above ChatPromptTemplate then values will
# not reflect. so instead we can use
chat_template = ChatPromptTemplate([
    ('system',  "You are helpful {domain} expert"),
    ('human',  "Explain in a simple terms, what is {topic}")
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Stumping'})

print(prompt)