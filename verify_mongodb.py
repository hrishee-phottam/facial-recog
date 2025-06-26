from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

# MongoDB Atlas configuration
username = 'phottam'
password = 'jWZLNOwfTAQVo7vQ'
db_name = 'phottam'
collection_name = 'people'

# Use the exact Atlas connection string format with properly escaped credentials
mongo_uri = f'mongodb+srv://{username}:{password}@cluster0.s35kdmn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

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

# Print some stats
print(f"\nCollection stats:")
print(f"Total documents: {collection.count_documents({})}")

# Print first document
print("\nFirst document:")
first_doc = collection.find_one()
if first_doc:
    print(first_doc)
else:
    print("No documents found!")
