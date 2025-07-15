from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(model = 'gpt-4', temperatue = 1.5, max_completion_tokens= 10) 
# Here temperature values sets the randomness or creativity of the response
# max_completion_tokens set the maximum number of toekns to be coming as output in response
result = llm.invoke("what is the name of lord budha?")
print(result)# gives the output with content and other stats like number of tokens
print(result.content)# gives the content only