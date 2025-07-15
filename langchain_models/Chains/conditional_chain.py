from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

result = classifier_chain.invoke({'feedback': 'This is a terrible smartphone'})

print(result)

prompt2 = PromptTemplate(
    template = 'Write an appropiate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = 'Write an appropiate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser ), # If
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser), #else
    RunnableLambda(lambda x: 'could not find sentiment') # Default condition
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is terrible phone'})

print(result)

chain.get_graph().print_ascii()