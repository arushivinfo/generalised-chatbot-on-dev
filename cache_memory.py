from pymongo import MongoClient

MONGO_URI = "localhost:27017"
DB_NAME = "mydatabase"
COLLECTION_NAME = "chat_history"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def save_to_cache(query, answer, user_id, team_id, session_id=None):
    """Save the latest query and answer to MongoDB chat_history for a user."""
    print(f"Saving to MongoDB: {user_id} | {query} -> {answer}")
    document = {
        "user_id": user_id, 
        "team_id": team_id, 
        "query": query, 
        "answer": answer
    }
    
    # Add session_id if provided
    if session_id:
        document["session_id"] = session_id
        print(f"Including session ID: {session_id}")
    
    collection.insert_one(document)

def get_last_memories(n=1, user_id=None, team_id=None, session_id=None):
    """Get last n memories (query-answer pairs) for a user from MongoDB."""
    if user_id is None:
        print("No user_id provided for memory retrieval.")
        return []
        
    # Build query
    query = {"user_id": user_id}
    if team_id is not None:
        query["team_id"] = team_id
    if session_id is not None:
        query["session_id"] = session_id
        
    cursor = collection.find(query).sort("_id", -1).limit(n)
    memories = list(cursor)[::-1]  # reverse so latest is last
    print(f"Loaded {len(memories)} cached memories for user {user_id}.")
    print("Cache content:", memories)
    return memories

def view_cache(user_id=None, team_id=None, session_id=None):
    """Return the full current cache for a user (for debugging)."""
    query = {}
    if user_id is not None:
        query["user_id"] = user_id
    if team_id is not None:
        query["team_id"] = team_id
    if session_id is not None:
        query["session_id"] = session_id
    
    cursor = collection.find(query).sort("_id", -1).limit(10)
    return list(cursor)

def get_memory_prompt(k=2, user_id=None, team_id=None, session_id=None):
    """Format the last k conversation turns for prompt context."""
    memories = get_last_memories(k, user_id, team_id, session_id)
    if not memories:
        return ""
        
    parts = []
    for memory in memories:
        parts.append(f"Human: {memory['query']}")
        parts.append(f"AI: {memory['answer']}")
    
    return "\n\n".join(parts)

def clear_cache(user_id=None, team_id=None, session_id=None):
    """Clear the cache for a specific user/team/session combination."""
    query = {}
    if user_id is not None:
        query["user_id"] = user_id
    if team_id is not None:
        query["team_id"] = team_id
    if session_id is not None:
        query["session_id"] = session_id
        
    result = collection.delete_many(query)
    print(f"Deleted {result.deleted_count} documents from cache.")
    return result.deleted_count

def get_unique_session_ids(user_id=None, team_id=None):
    """Get all unique session IDs for a user, optionally filtered by team_id."""
    query = {}
    if user_id is not None:
        query["user_id"] = user_id
    if team_id is not None:
        query["team_id"] = team_id
        
    # Only look for documents that have a session_id field
    query["session_id"] = {"$exists": True}
        
    # Use distinct to get unique session_ids
    unique_sessions = collection.distinct("session_id", query)
    
    # Return as list of session_id values
    return sorted(unique_sessions)

def load_chat_history(user_id, team_id, session_id):
    """Load all chat history for a specific session ID."""
    if not session_id:
        return []
        
    query = {
        "session_id": session_id
    }
    
    # Add user_id and team_id if provided
    if user_id:
        query["user_id"] = user_id
    if team_id:
        query["team_id"] = team_id
    
    # Get all chat messages for this session in chronological order
    cursor = collection.find(query).sort("_id", 1)
    
    # Convert to list of (role, content) tuples for chat history
    chat_history = []
    for doc in cursor:
        # Add user message
        chat_history.append(("user", doc["query"]))
        # Add assistant message
        chat_history.append(("assistant", doc["answer"]))
        
    return chat_history

def get_session_details(session_id):
    """Get details about a specific session."""
    if not session_id:
        return None
        
    query = {"session_id": session_id}
    
    # Find the first and last message of this session
    first_message = collection.find_one(query, sort=[("_id", 1)])
    last_message = collection.find_one(query, sort=[("_id", -1)])
    message_count = collection.count_documents(query)
    
    if not first_message or not last_message:
        return None
    
    # Extract timestamps from MongoDB ObjectIDs
    # Note: ObjectID contains timestamp of creation in first 4 bytes
    first_timestamp = first_message["_id"].generation_time
    last_timestamp = last_message["_id"].generation_time
    
    # Create a readable summary
    details = {
        "session_id": session_id,
        "message_count": message_count // 2,  # Divide by 2 because each query-answer pair is counted as 2
        "first_query": first_message["query"],
        "last_query": last_message["query"],
        "started": first_timestamp,
        "last_active": last_timestamp,
        "user_id": first_message.get("user_id", "unknown"),
        "team_id": first_message.get("team_id", "unknown")
    }
    
    return details

def rename_session(old_session_id, new_session_id):
    """Rename a session by updating its ID in all documents."""
    if not old_session_id or not new_session_id:
        return 0
    
    # Update all documents with the old session ID
    result = collection.update_many(
        {"session_id": old_session_id},
        {"$set": {"session_id": new_session_id}}
    )
    
    return result.modified_count