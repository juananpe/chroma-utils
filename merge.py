import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    print("OPENAI_API_KEY environment variable not found")
    sys.exit(1)

openai.api_key = openai_api_key
print("OPENAI_API_KEY is ready")

client = chromadb.HttpClient(host="localhost", port="8008")

embedding_function = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

if len(sys.argv) != 4:
    print("Usage: python merge_collections.py <collection1> <collection2> <merged_collection>")
    sys.exit(1)

collection1_name = sys.argv[1]
collection2_name = sys.argv[2]
merged_collection_name = sys.argv[3]

# Get the collections
collection1 = client.get_collection(name=collection1_name, embedding_function=embedding_function)
collection2 = client.get_collection(name=collection2_name, embedding_function=embedding_function)

# Create the new merged collection
merged_collection = client.create_collection(name=merged_collection_name, embedding_function=embedding_function)

# Add data from collection1 to the merged collection
data1 = collection1.get()
if data1["ids"]:
    merged_collection.add(
        embeddings=data1["embeddings"],
        documents=data1["documents"],
        metadatas=data1["metadatas"],
        ids=data1["ids"]
    )

# Add data from collection2 to the merged collection
data2 = collection2.get()
if data2["ids"]:
    merged_collection.add(
        embeddings=data2["embeddings"],
        documents=data2["documents"],
        metadatas=data2["metadatas"],
        ids=data2["ids"]
    )

print(f"Collections '{collection1_name}' and '{collection2_name}' merged into '{merged_collection_name}'.")