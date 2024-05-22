import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import openai
import os
import json
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# Get the collection name from the first command-line argument
if len(sys.argv) > 2:
    collection_name = sys.argv[1]
    filename = sys.argv[2]
else:
    print("Usage: python import.py <collection_name> <json_file_name>")
    sys.exit(1)

client = chromadb.HttpClient(host="localhost", port="8008")

# Get the OpenAI API key from the environment variable or .env file
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is not None:
    openai.api_key = openai_api_key
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
    sys.exit(1)

embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-3-small"
)


# Create the new collection
new_collection = client.create_collection(
    name=collection_name, embedding_function=embedding_function
)

# Load data from JSON
with open(filename, "r") as f:
    data = json.load(f)

# Add data to the new collection
if data["ids"]:
    new_collection.add(
        embeddings=data["embeddings"],
        documents=data["documents"],
        metadatas=data["metadatas"],
        ids=data["ids"],
    )

print(f"Data imported from '{filename}' into collection '{collection_name}'.")
