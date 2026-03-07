from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
user_docs = ["Taj Mahal is in agra, India.",
            "the Eiffel Tower is in Paris, France.",
            "The Colosseum is in Rome, Italy."]
user_query = "Where is the Eiffel Tower?"
user_docs_embedding = emb.embed_documents(user_docs)
user_query_embedding = emb.embed_query(user_query)
from sklearn.metrics.pairwise import cosine_similarity
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print("------- start-----")
print("Question: ", user_query)
print("LLM Result: ", user_docs[index])
print("confidance: ", score)
print ("-----END----")