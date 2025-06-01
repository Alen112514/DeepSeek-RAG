import chromadb

# Use the path to your 'DB/image' directory (where those files live)
client = chromadb.PersistentClient(path="/Users/l/Desktop/LeapLead/RAG/DB/image")
print("Collections:", client.list_collections())
collection = client.get_or_create_collection("langchain")
results = collection.get(include=['embeddings', 'metadatas', 'documents'])

import numpy as np
for emb in results['embeddings']:
    print(np.linalg.norm(emb))

# for idx in range(len(results['ids'])):
#     print(f"ID: {results['ids'][idx]}")
#     print(f"Embedding (first 10 dims): {results['embeddings'][idx][:10]}")
#     print(f"Metadata: {results['metadatas'][idx]}")
#     print(f"Document: {results['documents'][idx]}\n")

