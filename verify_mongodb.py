import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

# Required environment variables
required_vars = [
    'MONGODB_URI',
    'MONGODB_DB_NAME',

def load_environment():
    """Load and validate required environment variables."""
    load_dotenv()
    
    required_vars = [
        'MONGODB_URI',
        'MONGODB_DB_NAME',
        'MONGODB_COLLECTION_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    return {
        'mongodb_uri': os.getenv('MONGODB_URI'),
        'db_name': os.getenv('MONGODB_DB_NAME'),
        'collection_name': os.getenv('MONGODB_COLLECTION_NAME')
    }

def connect_to_mongodb(uri, db_name):
    """Establish connection to MongoDB and return database object."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Force a connection check
        client.admin.command('ping')
        db = client[db_name]
        logging.info("✅ Successfully connected to MongoDB")
        return client, db
    except ConnectionFailure as e:
        logging.error(f"❌ Failed to connect to MongoDB: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

def check_collection(db, collection_name):
    """Check if collection exists and return collection object."""
    collection_names = db.list_collection_names()
    
    if collection_name not in collection_names:
        logging.warning(f"⚠️  Collection '{collection_name}' does not exist")
        return None
        
    collection = db[collection_name]
    logging.info(f"📊 Found collection: {collection_name}")
    return collection

def analyze_collection(collection):
    """Analyze collection and print statistics."""
    try:
        # Count total documents
        total_docs = collection.count_documents({})
        logging.info(f"📄 Total documents: {total_docs}")
        
        if total_docs == 0:
            logging.warning("⚠️  No documents found in the collection")
            return
            
        # Get sample documents
        sample = collection.find_one()
        logging.info("\n🔍 Sample document structure:")
        pprint(sample, indent=2)
        
        # Count documents with embeddings
        docs_with_embeddings = collection.count_documents({"embedding": {"$exists": True, "$ne": None}})
        logging.info(f"\n📊 Documents with embeddings: {docs_with_embeddings}/{total_docs} ({docs_with_embeddings/total_docs:.1%})")
        
        # Count documents with box coordinates
        docs_with_box = collection.count_documents({"box": {"$exists": True, "$ne": None}})
        logging.info(f"📊 Documents with box coordinates: {docs_with_box}/{total_docs} ({docs_with_box/total_docs:.1%})")
        
        # Show a few sample embeddings
        if docs_with_embeddings > 0:
            logging.info("\n🔎 Sample embedding (first 5 values):")
            sample_with_embedding = collection.find_one({"embedding": {"$exists": True, "$ne": None}})
            if sample_with_embedding and 'embedding' in sample_with_embedding:
                embedding = sample_with_embedding['embedding']
                if isinstance(embedding, list) and len(embedding) > 5:
                    logging.info(f"Embedding length: {len(embedding)}")
                    logging.info(f"Sample values: {embedding[:5]}...")
        
        # Show a sample with box coordinates
        if docs_with_box > 0:
            logging.info("\n📏 Sample box coordinates:")
            sample_with_box = collection.find_one({"box": {"$exists": True, "$ne": None}})
            if sample_with_box and 'box' in sample_with_box:
                pprint(sample_with_box['box'], indent=2)
                
    except Exception as e:
        logging.error(f"❌ Error analyzing collection: {e}")

def main():
    """Main function to verify MongoDB connection and data."""
    try:
        # Load environment
        env = load_environment()
        
        # Connect to MongoDB
        client, db = connect_to_mongodb(env['mongodb_uri'], env['db_name'])
        
        # Check collection
        collection = check_collection(db, env['collection_name'])
        
        if collection is not None:
            # Analyze collection
            analyze_collection(collection)
        
    except Exception as e:
        logging.error(f"❌ An error occurred: {e}", exc_info=True)
        return 1
    finally:
        if 'client' in locals():
            client.close()
            logging.info("🔌 Closed MongoDB connection")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
else:
    print("No documents found!")
