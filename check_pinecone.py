import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")

try:
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print("Indexes found:")
    for i in indexes:
        print(f"- {i.name} (Status: {i.status['state']})")
    
    if not indexes:
        print("No indexes found on this account.")

except Exception as e:
    print(f"Error: {e}")
