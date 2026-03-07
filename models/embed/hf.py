from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "I love machine learning",
    "Embeddings convert text into vectors",
    "Sentence transformers are powerful"
]

# Encode sentences
embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("First embedding:", embeddings[0])
