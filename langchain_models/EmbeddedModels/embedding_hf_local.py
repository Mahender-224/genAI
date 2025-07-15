from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# for query
# text = "Hyderabad is nawab's city"

# vector = embedding.embed_query(text)

# print(str(vector))

# for documents
documents = [
    "Delhi is the capital of india",
    "Kolkate is the beautiful but good city"
]

vectors = embedding.embed_documents(documents)

print(str(vectors))