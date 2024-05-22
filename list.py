import chromadb

# Configure ChromaDB client
client = chromadb.HttpClient(host="localhost", port="8008")

collection_names = client.list_collections()

print(collection_names)

