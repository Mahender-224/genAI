from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 300)

documents = [
    "Virat Kohli is an indian cricketer known for his aggressive batting and leadership",
    "MS Dhoni is a former indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar, also known as 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

print(scores) # print the scores, highest matching scores of a document can be known.
