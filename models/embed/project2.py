from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
user_docs = [ "Climate change is mainly caused by greenhouse gases released from burning fossil fuels and industrial activities.",
 "The effects of climate change include rising global temperatures, melting glaciers, and increasing sea levels.",
"Climate change leads to extreme weather events such as heatwaves, floods, droughts, and stronger storms.", 
"Climate change disrupts ecosystems, damages agriculture, and increases risks to human health worldwide.",
 "Using renewable energy, planting trees, and reducing emissions are key solutions to slow climate change.",
"Climate change causes rising temperatures, melting ice caps, and sea level rise, impacting life globally.",
 "Global warming results in extreme weather like droughts, floods, and heatwaves, which are major climate impacts.",
 "Reducing carbon emissions through clean energy and sustainability can help control climate change effects." ]
user_query = "What are the effects of climate change?"
user_docs_embedding = emb.embed_documents(user_docs)
user_query_embedding = emb.embed_query(user_query)

from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity( [user_query_embedding], user_docs_embedding)[0]
indexed_scores = list(enumerate(scores))

filtered = [(i, s) for i, s in indexed_scores if s > 0.55]
sorted_results = sorted(filtered, key=lambda x: x[1], reverse=True)
top3 = sorted_results[:3]
print("------- start-----")
print("----- Top 3 Results -----")
for index, score in top3:
    print(f"Document: {user_docs[index]}")
    print(f"Similarity: {score:.4f}")
print ("-----END----")