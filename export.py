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
if len(sys.argv) > 1:
    collection_name = sys.argv[1]
else:
    print("Error: Collection name is required as the first command-line argument")
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

collection = client.get_collection(
    name=collection_name, embedding_function=embedding_function
)
data = collection.get( include=['embeddings', 'documents', 'metadatas'] )

# Save the collection data to a JSON file
with open(f"{collection_name}_data.json", "w") as f:
    json.dump(data, f)

print(
    f"Data from collection '{collection_name}' dumped to '{collection_name}_data.json'."
)
