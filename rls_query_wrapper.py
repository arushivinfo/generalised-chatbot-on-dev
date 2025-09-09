"""
RLS Query Wrapper - Automatically applies Row-Level Security to MongoDB queries
"""

from typing import Dict, List, Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from rls_engine import create_rls_interceptor
from auth_manager import get_current_user_context, get_current_user_id
import logging

logger = logging.getLogger(__name__)

class RLSMongoClient:
    """MongoDB client wrapper that automatically applies RLS filtering."""
    
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.connection_string = connection_string
        self.database_name = database_name
    
    def get_collection(self, collection_name: str) -> 'RLSCollection':
        """Get a collection with RLS protection."""
        return RLSCollection(self.db[collection_name], collection_name)

class RLSCollection:
    """Collection wrapper that applies RLS to all operations."""
    
    def __init__(self, collection: Collection, collection_name: str):
        self.collection = collection
        self.collection_name = collection_name
        self._rls_interceptor = None
    
    def _get_rls_interceptor(self):
        """Get or create RLS interceptor for current user."""
        user_context = get_current_user_context()
        if not user_context:
            raise PermissionError("No authenticated user found. Please log in.")
        
        if not self._rls_interceptor:
            self._rls_interceptor = create_rls_interceptor(
                user_context.user_id, 
                user_context.role, 
                user_context.permissions
            )
        
        return self._rls_interceptor
    
    def find(self, filter_query: Dict = None, *args, **kwargs):
        """Find documents with RLS filtering."""
        filter_query = filter_query or {}
        rls = self._get_rls_interceptor()
        
        # Apply RLS filtering
        enhanced_query = rls.enhance_find_query(self.collection_name, filter_query)
        
        logger.info(f"RLS Find: {self.collection_name} - Original: {filter_query}, Enhanced: {enhanced_query}")
        
        return self.collection.find(enhanced_query, *args, **kwargs)
    
    def find_one(self, filter_query: Dict = None, *args, **kwargs):
        """Find one document with RLS filtering."""
        filter_query = filter_query or {}
        rls = self._get_rls_interceptor()
        
        enhanced_query = rls.enhance_find_query(self.collection_name, filter_query)
        
        logger.info(f"RLS FindOne: {self.collection_name} - Enhanced: {enhanced_query}")
        
        return self.collection.find_one(enhanced_query, *args, **kwargs)
    
    def aggregate(self, pipeline: List[Dict], *args, **kwargs):
        """Aggregate with RLS filtering."""
        rls = self._get_rls_interceptor()
        
        enhanced_pipeline = rls.enhance_aggregate_pipeline(self.collection_name, pipeline)
        
        logger.info(f"RLS Aggregate: {self.collection_name} - Original: {pipeline}, Enhanced: {enhanced_pipeline}")
        
        return self.collection.aggregate(enhanced_pipeline, *args, **kwargs)
    
    def update_one(self, filter_query: Dict, update_doc: Dict, *args, **kwargs):
        """Update one document with RLS filtering."""
        rls = self._get_rls_interceptor()
        
        enhanced_filter, enhanced_update = rls.enhance_update_query(
            self.collection_name, filter_query, update_doc
        )
        
        logger.info(f"RLS UpdateOne: {self.collection_name} - Filter: {enhanced_filter}")
        
        return self.collection.update_one(enhanced_filter, enhanced_update, *args, **kwargs)
    
    def update_many(self, filter_query: Dict, update_doc: Dict, *args, **kwargs):
        """Update many documents with RLS filtering."""
        rls = self._get_rls_interceptor()
        
        enhanced_filter, enhanced_update = rls.enhance_update_query(
            self.collection_name, filter_query, update_doc
        )
        
        logger.info(f"RLS UpdateMany: {self.collection_name} - Filter: {enhanced_filter}")
        
        return self.collection.update_many(enhanced_filter, enhanced_update, *args, **kwargs)
    
    def delete_one(self, filter_query: Dict, *args, **kwargs):
        """Delete one document with RLS filtering."""
        rls = self._get_rls_interceptor()
        
        enhanced_query = rls.enhance_delete_query(self.collection_name, filter_query)
        
        logger.info(f"RLS DeleteOne: {self.collection_name} - Filter: {enhanced_query}")
        
        return self.collection.delete_one(enhanced_query, *args, **kwargs)
    
    def delete_many(self, filter_query: Dict, *args, **kwargs):
        """Delete many documents with RLS filtering."""
        rls = self._get_rls_interceptor()
        
        enhanced_query = rls.enhance_delete_query(self.collection_name, filter_query)
        
        logger.info(f"RLS DeleteMany: {self.collection_name} - Filter: {enhanced_query}")
        
        return self.collection.delete_many(enhanced_query, *args, **kwargs)
    
    def count_documents(self, filter_query: Dict = None, *args, **kwargs):
        """Count documents with RLS filtering."""
        filter_query = filter_query or {}
        rls = self._get_rls_interceptor()
        
        enhanced_query = rls.enhance_find_query(self.collection_name, filter_query)
        
        return self.collection.count_documents(enhanced_query, *args, **kwargs)

# Factory function for easy integration
def get_rls_database(connection_string: str, database_name: str) -> RLSMongoClient:
    """Get a MongoDB client with RLS protection."""
    return RLSMongoClient(connection_string, database_name)

# Example usage
def example_rls_queries():
    """Example of how to use RLS-protected queries."""
    
    # Get RLS-protected database
    rls_client = get_rls_database("mongodb://localhost:27017", "mydb")
    users_collection = rls_client.get_collection("users")
    
    # All these queries will automatically have RLS applied
    
    # Find - will add user_id filter automatically
    users = list(users_collection.find({"status": "active"}))
    
    # Find one
    user = users_collection.find_one({"email": "user@example.com"})
    
    # Aggregate - will add RLS to $match stages
    pipeline = [
        {"$match": {"department": "sales"}},
        {"$group": {"_id": "$role", "count": {"$sum": 1}}}
    ]
    results = list(users_collection.aggregate(pipeline))
    
    # Update - will only update user's own records
    users_collection.update_one(
        {"email": "user@example.com"}, 
        {"$set": {"last_login": "2024-01-01"}}
    )
    
    # Delete - will only delete user's own records
    users_collection.delete_one({"temp_flag": True})
    
    return users, user, results
