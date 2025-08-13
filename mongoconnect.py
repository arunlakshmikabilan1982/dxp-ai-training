from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# MongoDB URI (modify if needed)
uri = "mongodb://localhost:27017/"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)  # 3 second timeout
    client.admin.command("ping")  # Test connection
    print("✅ Connected to MongoDB successfully!")

    # Optional: Show all databases
    print("📂 Databases:", client.list_database_names())

except ConnectionFailure as e:
    print("❌ Could not connect to MongoDB:", e)
