from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


em = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

docs =[
    "paris, France-on the champ de Mars in the 7th arrondissement, near the Sein.",
"The Eiffel tower is in paris, france."
]

result = em.embed_documents(docs)

print(result)