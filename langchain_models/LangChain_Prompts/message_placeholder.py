from langchain_core.prompt import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful customer support agent'),
    MessagesPlaceholder(variable_name = 'chat_history')
    ('human', '{query}')
])

chat_history = []
# Load chat history from chat_history.txt
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_template.invoke({'chat_history':chat_history, 'query': 'Where is my refund'})

print(prompt)