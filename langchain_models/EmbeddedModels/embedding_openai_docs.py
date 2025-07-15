from langcahin_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)

documents = [
    "Delhi is the capital of india",
    "Kolkate is the beautiful but good city",
    "Paris has world biggest tower"
]

result = embedding.embed_documents(documents)
print(str(result))