from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# 1
# while True:
#     user_input = input("user: ")
#     if user_input == 'exit':
#         break
#     result = model.invoke(user_input)
#     print("AI: ", result.content)

# here it is simple chatbot, but the issue with this chatbot is
# it can't remember the history of chat
# Now to remeber all the chats, we create a memory
# Modified above while loop is

# 2
# chat_history = []

# while True:
#     user_input = input("user: ")
#     chat_history.append(user_input)
#     if user_input == 'exit':
#         break
#     result = model.invoke(user_input)
#     chat_history.append(result.content)
#     print("AI: ", result.content)

# print(chat_history)

# Now the problem with this approach is we are storing the 
# inputs and responses directly into chat_history, but model don't know 
# which is user_input and which generated response.
# for that langchain has

# 3
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_history = [
    SystemMessage(content="You are helpful AI assistant")
]

while True:
    user_input = input("user: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ", result.content)

print(chat_history)