import sys
import chromadb

# Configure ChromaDB client
client = chromadb.HttpClient(host="localhost", port="8008")

# Get the collection name from the command line argument
if len(sys.argv) != 2:
    print("Usage: python delete_collection.py <collection_name>")
    sys.exit(1)

collection_name = sys.argv[1]

# Delete the collection
client.delete_collection(collection_name)

print(f"Collection '{collection_name}' deleted successfully!")