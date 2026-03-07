
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vec = embeddings.embed_query("Where is the Eiffil Tower")

print(len(vec))
