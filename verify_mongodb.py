import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

# Required environment variables
required_vars = [
    'MONGODB_URI',
    'MONGODB_DB_NAME',
    'MONGODB_COLLECTION_NAME'
]

# Check for missing required environment variables
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Load configuration from environment variables
mongo_uri = os.environ['MONGODB_URI']
db_name = os.environ['MONGODB_DB_NAME']
collection_name = os.environ['MONGODB_COLLECTION_NAME']

# Create client with Server API version 1
client = MongoClient(mongo_uri, server_api=ServerApi('1'))

# Verify connection
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {str(e)}")
    exit(1)

# Get database and collection
db = client[db_name]
collection = db[collection_name]

# Count documents
count = collection.count_documents({})
print(f"Total documents in collection: {count}")

# Show sample documents
if count > 0:
    print("\nSample documents:")
    for doc in collection.find().limit(3):
        print("\nDocument:")
        print(f"  ID: {doc['_id']}")
        print(f"  Filename: {doc.get('filename', 'N/A')}")
        print(f"  Embedding length: {len(doc.get('embedding', []))}")
        
        # Show box coordinates if available
        if 'box' in doc:
            print(f"  Face box: x={doc['box'].get('x_min')}-{doc['box'].get('x_max')}, "
                  f"y={doc['box'].get('y_min')}-{doc['box'].get('y_max')}, "
                  f"probability={doc['box'].get('probability', 'N/A')}")
        
        # Show execution time if available
        if 'execution_time' in doc:
            print(f"  Execution time: {doc['execution_time']}")
else:
    print("No documents found in the collection.")

# Print first document
print("\nFirst document:")
first_doc = collection.find_one()
if first_doc:
    print(first_doc)
else:
    print("No documents found!")
