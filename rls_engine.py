"""
Row-Level Security (RLS) Engine for MongoDB
Automatically filters queries based on user ownership and permissions.
"""
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from schema_registry import (
    get_rls_config, get_all_fields, load_registry, 
    get_collection_relations, is_valid_rls_field, get_connection_config
)
from datetime import datetime
from llm_services import call_admin_ai_model
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    """User context for RLS filtering."""
    user_id: str
    role: str
    permissions: List[str] = None
    bypass_rls: bool = False
    team_id: Optional[str] = None
    session_id: Optional[str] = None
    patient_id: Optional[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []

class RLSFieldDetector:
    """AI and heuristic-based field detection for RLS."""
    
    OWNERSHIP_PATTERNS = {
        "exact_matches": [
            "user_id", "customer_id", "employee_id", "owner_id", "account_id", "member_id", 
            "patient_id", "doctor_id", "tenant_id", "organization_id", "company_id", "project_id",
            "student_id", "teacher_id", "vendor_id", "supplier_id", "client_id", "partner_id"
        ],
        "suffix_patterns": [
            "_user_id", "_customer_id", "_owner_id", "_account_id", "_member_id", "_patient_id", 
            "_doctor_id", "_tenant_id", "_org_id", "_company_id", "_project_id", "_student_id",
            "_teacher_id", "_vendor_id", "_supplier_id", "_client_id", "_partner_id"
        ],
        "contextual_patterns": [
            "user.*_id", "customer.*_id", "owner.*_id", "account.*_id", "member.*_id", 
            "patient.*_id", "doctor.*_id", "tenant.*_id", "org.*_id", "company.*_id",
            "project.*_id", "student.*_id", "teacher.*_id", "vendor.*_id", "supplier.*_id"
        ]
    }
    
    def __init__(self):
        pass
    
    def detect_ownership_field(self, collection: str, fields: List[Dict]) -> Optional[str]:
        """Detect the best ownership field for a collection."""
        field_names = [f["name"] for f in fields]
        
        # 1. Try exact matches first
        for field in field_names:
            if field.lower() in [p.lower() for p in self.OWNERSHIP_PATTERNS["exact_matches"]]:
                logger.info(f"RLS: Found exact match field '{field}' for collection '{collection}'")
                return field
        
        # 2. Try suffix patterns
        for field in field_names:
            for pattern in self.OWNERSHIP_PATTERNS["suffix_patterns"]:
                if field.lower().endswith(pattern.lower().replace("*", "")):
                    logger.info(f"RLS: Found suffix match field '{field}' for collection '{collection}'")
                    return field
        
        # 3. Try contextual patterns (regex)
        for field in field_names:
            for pattern in self.OWNERSHIP_PATTERNS["contextual_patterns"]:
                regex_pattern = pattern.replace("*", ".*").replace("_id", r".*_id$")
                if re.match(regex_pattern, field.lower()):
                    logger.info(f"RLS: Found contextual match field '{field}' for collection '{collection}'")
                    return field
        
        # 4. Try AI detection
        try:
            ai_suggestion = self._ai_detect_field(collection, fields)
            if ai_suggestion:
                logger.info(f"RLS: AI suggested field '{ai_suggestion}' for collection '{collection}'")
                return ai_suggestion
        except Exception as e:
            logger.warning(f"AI field detection failed: {e}")
        
        logger.warning(f"RLS: No suitable ownership field found for collection '{collection}'")
        return None
    
    def _ai_detect_field(self, collection: str, fields: List[Dict]) -> Optional[str]:
        """Use AI to detect the best ownership field."""
        field_info = []
        for field in fields:
            field_info.append({
                "name": field["name"],
                "type": field["type"],
                "description": field.get("description", ""),
                "sample_options": field.get("options", [])[:5]  # First 5 options
            })
        
        prompt = f"""
You are analyzing a MongoDB collection for Row-Level Security implementation.

Collection: {collection}
Fields: {json.dumps(field_info, indent=2)}

TASK: Identify the BEST field for user ownership filtering (Row-Level Security).

RULES:
1. Field should uniquely identify which user/customer/account owns each document
2. Prefer fields ending with '_id' that represent users, customers, employees, owners
3. Look for semantic patterns: user, customer, employee, owner, account, member
4. Consider field descriptions and sample values
5. Return ONLY the field name, nothing else
6. If no suitable field exists, return "null"

Examples of good ownership fields:
- user_id, customer_id, employee_id, owner_id, account_id, member_id
- created_by, assigned_to, belongs_to (if they reference users)

RESPOND WITH ONLY THE FIELD NAME OR "null":
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response_content = call_admin_ai_model(messages)
            suggested_field = response_content.strip().strip('"').strip("'")
            
            # Validate the suggestion
            field_names = [f["name"] for f in fields]
            if suggested_field in field_names and suggested_field != "null":
                return suggested_field
            
        except Exception as e:
            logger.error(f"AI field detection error: {e}")
        
        return None

class RLSQueryInterceptor:
    def __init__(self, user_context: UserContext, rls_config: Dict = None):
        self.user = user_context
        self.config = rls_config or get_rls_config()
        self.active_layers = self._get_active_layers()
        self.audit_log = []  # ✅ Initialize audit log first
        self.field_detector = RLSFieldDetector()  # Initialize field detector
    
    def _get_active_layers(self) -> List[Dict]:
        """Get enabled RLS layers sorted by order."""
        layers = self.config.get("rls_layers", [])
        return sorted([layer for layer in layers if layer.get("enabled", False)], 
                     key=lambda x: x.get("order", 999))
    
    def should_bypass_rls(self) -> bool:
        """Check if RLS should be bypassed for current user."""
        if not self.config.get("enabled", False):
            return True
        
        if self.user.bypass_rls:
            return True
        
        bypass_roles = self.config.get("bypass_roles", [])
        if self.user.role in bypass_roles:
            self._audit_log("RLS_BYPASS", f"User role '{self.user.role}' bypasses RLS")
            return True
        
        return False
    
    def resolve_ownership_field(self, collection: str) -> Optional[str]:
        """Resolve the ownership field for a collection."""
        # 1. Check explicit collection override
        collections_config = self.config.get("collections", {})
        if collection in collections_config:
            field = collections_config[collection].get("user_field")
            if field:
                return field
        
        # 2. Try auto-detection
        try:
            reg = load_registry()
            all_fields = get_all_fields(reg)
            if collection in all_fields:
                detected_field = self.field_detector.detect_ownership_field(
                    collection, all_fields[collection]
                )
                if detected_field:
                    return detected_field
        except Exception as e:
            logger.error(f"Field detection error for {collection}: {e}")
        
        # 3. Fallback to default
        default_field = self.config.get("default_user_field", "user_id")
        logger.info(f"RLS: Using default field '{default_field}' for collection '{collection}'")
        return default_field
    
    def enhance_find_query(self, collection: str, query: Dict) -> Dict:
        """Apply multi-layer RLS with per-layer enforcement modes."""
        if self.should_bypass_rls():
            return query
        
        enhanced_query = query.copy()
        
        for layer in self.active_layers:
            field_name = layer["field_name"]
            enforcement_mode = layer.get("enforcement_mode", "base_only")
            
            # Get user value for this layer
            user_value = getattr(self.user, field_name, None)
            if user_value and (layer.get("required", False) or user_value):
                enhanced_query[field_name] = user_value
                self._audit_log("RLS_LAYER_APPLIED", f"Applied layer {field_name}={user_value} to {collection}")
        
        return enhanced_query
    
    def _audit_log(self, event_type: str, message: str):
        """Add an entry to the audit log."""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "user_id": self.user.user_id,
            "role": self.user.role
        })
    
    def enhance_find_query_with_multilayer(self, collection: str, query: Dict, rls_values: Dict = None) -> Dict:
        """Enhanced find query with multi-layer RLS filtering and relationship traversal."""
        return self.enhance_find_query_with_relationship_traversal(collection, query, rls_values)
    
    def enhance_find_query_with_relationship_traversal(self, collection: str, query: Dict, rls_values: Dict = None) -> Dict:
        """Enhanced find query with multi-layer RLS filtering and relationship traversal."""
        if self.should_bypass_rls():
            return query
        
        # Get configured RLS layers
        rls_layers = self.config.get("rls_layers", [])
        if not rls_layers and not rls_values:
            # Fall back to single-layer RLS
            return self.enhance_find_query(collection, query)
        
        # Check which fields are available in this collection
        try:
            reg = load_registry()
            all_fields = get_all_fields(reg)
            collection_fields_info = all_fields.get(collection, [])
            
            # Get field names that are suitable for RLS
            available_fields = []
            field_types = {}
            for field_info in collection_fields_info:
                if is_valid_rls_field(field_info):
                    field_name = field_info["name"]
                    available_fields.append(field_name)
                    field_types[field_name] = field_info.get("type")
                    
        except Exception as e:
            logger.error(f"Error checking collection fields for {collection}: {e}")
            available_fields = []
            field_types = {}
        
        enhanced_query = query.copy()
        rls_filters = []
        
        # Apply each RLS layer 
        for layer in rls_layers:
            if not layer.get("enabled", True):
                continue
                
            field_name = layer.get("field_name")
            enforcement_mode = layer.get("enforcement_mode", "base_only")
            
            if not field_name:
                continue
                
            # Get value from rls_values or use user context
            field_value = None
            if rls_values and field_name in rls_values:
                field_value = rls_values[field_name]
                # Only use the value if it's not empty/whitespace  
                if not field_value or not str(field_value).strip():
                    field_value = None
                    self._audit_log("RLS_SKIP", f"Field '{field_name}' has empty/null value - skipping layer")
                    continue
            elif field_name == "user_id" and hasattr(self.user, 'user_id') and self.user.user_id not in ["anonymous", None, ""]:
                field_value = self.user.user_id
                
            if field_value is None or not str(field_value).strip():
                continue
                
            # Check if field exists directly in this collection
            field_exists_directly = field_name in available_fields
            
            if field_exists_directly:
                # Direct field application (traditional RLS)
                field_type = field_types.get(field_name, "string")
                if field_type in ["int", "integer", "number"]:
                    try:
                        field_value = int(field_value)
                    except Exception:
                        pass  # Keep as string if conversion fails
                
                rls_filters.append({field_name: field_value})
                self._audit_log("RLS_LAYER_APPLIED", f"Applied direct RLS layer {field_name}={field_value} to {collection}")
                
            elif enforcement_mode == "all_involved":
                # Relationship-based RLS - this is the key enhancement
                try:
                    relationship_filter = self._build_relationship_filter(collection, field_name, field_value, reg)
                    if relationship_filter:
                        rls_filters.append(relationship_filter)
                        self._audit_log("RLS_LAYER_APPLIED", f"Applied relationship RLS layer {field_name}={field_value} to {collection}")
                    else:
                        # If no relationship found, deny access for security
                        self._audit_log("RLS_DENY", f"No relationship path found for {field_name} in {collection} - denying access")
                        rls_filters.append({"_id": {"$in": []}})
                except Exception as e:
                    logger.error(f"Error building relationship filter for {field_name} in {collection}: {e}")
                    # On error with all_involved, be restrictive
                    rls_filters.append({"_id": {"$in": []}})
            else:
                # base_only mode - skip fields that don't exist in collection
                self._audit_log("RLS_SKIP", f"Field '{field_name}' not found in collection '{collection}' and enforcement is base_only - skipping")
        
        # Combine RLS filters with base query
        if rls_filters:
            if len(rls_filters) == 1:
                rls_combined = rls_filters[0]
            else:
                rls_combined = {"$and": rls_filters}
            
            if enhanced_query:
                enhanced_query = {"$and": [enhanced_query, rls_combined]}
            else:
                enhanced_query = rls_combined
        else:
            # If no RLS filters applied and we have RLS values, log warning
            if rls_values:
                self._audit_log("RLS_WARNING", f"No RLS filters applied for collection {collection} with values {rls_values}")
        
        return enhanced_query
    
    def _build_relationship_filter(self, collection: str, field_name: str, field_value: str, reg: Dict) -> Optional[Dict]:
        """Build a MongoDB filter using relationship traversal for RLS enforcement.
        
        This is completely dynamic - works with ANY collection schema and relationship structure.
        No hardcoded healthcare/business logic.
        """
        try:
            # Get database connection
            cfg = get_connection_config(reg)
            if not cfg.get("mongo_uri") or not cfg.get("mongo_db"):
                logger.error("Database connection not configured")
                return None
                
            client = MongoClient(cfg["mongo_uri"])
            db = client[cfg["mongo_db"]]

            # Get collection relationships from schema registry
            relations = get_collection_relations(reg)
            
            # Find path from field_name to target collection through relationships
            target_ids = self._find_related_ids_through_relationships(
                db, collection, field_name, field_value, relations
            )
            
            if target_ids:
                # Return filter that matches documents with these IDs
                return {"_id": {"$in": target_ids}}
            else:
                # No related records found
                return {"_id": {"$in": []}}
                
        except Exception as e:
            logger.error(f"Error in _build_relationship_filter: {e}")
            return None

    def _find_related_ids_through_relationships(self, db, target_collection: str, field_name: str, field_value: str, relations: Dict) -> List:
        """
        Find IDs in target_collection that are related to field_value through collection relationships.
        Completely dynamic - no hardcoded business logic.
        """
        try:
            # Strategy: Use MongoDB aggregation to traverse relationships
            # 1. Find intermediate collections that have both field_name and link to target_collection
            
            for collection_name, collection_relations in relations.items():
                try:
                    # Check if this collection has the field we're filtering on
                    if field_name in [rel.get("local_field") for rel in collection_relations]:
                        # This collection has our field, now check if it relates to target_collection
                        for relation in collection_relations:
                            if relation.get("foreign_collection") == target_collection:
                                # Found a path! Get the linking IDs
                                intermediate_docs = list(db[collection_name].find(
                                    {field_name: field_value},
                                    {relation["local_field"]: 1}
                                ))
                                
                                if intermediate_docs:
                                    linking_ids = [doc.get(relation["local_field"]) for doc in intermediate_docs if doc.get(relation["local_field"])]
                                    
                                    # Now find target documents
                                    target_docs = list(db[target_collection].find(
                                        {relation["foreign_field"]: {"$in": linking_ids}},
                                        {"_id": 1}
                                    ))
                                    
                                    return [doc["_id"] for doc in target_docs]
                
                except Exception as e:
                    logger.warning(f"Error checking relationship path through {collection_name}: {e}")
                    continue
            
            return []
            
        except Exception as e:
            logger.error(f"Error in _find_related_ids_through_relationships: {e}")
            return []
        """Enhanced find query with multi-layer RLS filtering and relationship traversal."""
        if self.should_bypass_rls():
            return query
        
        # Get configured RLS layers
        rls_layers = self.config.get("rls_layers", [])
        if not rls_layers and not rls_values:
            # Fall back to single-layer RLS
            return self.enhance_find_query(collection, query)
        
        # Check which fields are available in this collection
        try:
            reg = load_registry()
            all_fields = get_all_fields(reg)
            collection_fields_info = all_fields.get(collection, [])
            
            # Get field names that are suitable for RLS
            available_fields = []
            field_types = {}
            for field_info in collection_fields_info:
                if is_valid_rls_field(field_info):
                    field_name = field_info["name"]
                    available_fields.append(field_name)
                    field_types[field_name] = field_info.get("type")
                    
        except Exception as e:
            logger.error(f"Error checking collection fields for {collection}: {e}")
            available_fields = []
            field_types = {}
        
        enhanced_query = query.copy()
        rls_filters = []
        
        # Apply each RLS layer 
        for layer in rls_layers:
            if not layer.get("enabled", True):
                continue
                
            field_name = layer.get("field_name")
            enforcement_mode = layer.get("enforcement_mode", "base_only")
            
            if not field_name:
                continue
                
            # Get value from rls_values or use user context
            field_value = None
            if rls_values and field_name in rls_values:
                field_value = rls_values[field_name]
                # Only use the value if it's not empty/whitespace  
                if not field_value or not str(field_value).strip():
                    field_value = None
                    self._audit_log("RLS_SKIP", f"Field '{field_name}' has empty/null value - skipping layer")
                    continue
            elif field_name == "user_id" and hasattr(self.user, 'user_id') and self.user.user_id not in ["anonymous", None, ""]:
                field_value = self.user.user_id
                
            if field_value is None or not str(field_value).strip():
                continue
                
            # Check if field exists directly in this collection
            field_exists_directly = field_name in available_fields
            
            if field_exists_directly:
                # Direct field application (traditional RLS)
                field_type = field_types.get(field_name, "string")
                if field_type in ["int", "integer", "number"]:
                    try:
                        field_value = int(field_value)
                    except Exception:
                        pass  # Keep as string if conversion fails
                
                rls_filters.append({field_name: field_value})
                self._audit_log("RLS_LAYER_APPLIED", f"Applied direct RLS layer {field_name}={field_value} to {collection}")
                
            elif enforcement_mode == "all_involved":
                # Relationship-based RLS - this is the key enhancement
                try:
                    relationship_filter = self._build_relationship_filter(collection, field_name, field_value, reg)
                    if relationship_filter:
                        rls_filters.append(relationship_filter)
                        self._audit_log("RLS_RELATIONSHIP_APPLIED", f"Applied relationship-based RLS for {field_name}={field_value} to {collection}")
                    else:
                        # For all_involved mode, if no relationship path found, return empty result
                        self._audit_log("RLS_NO_RELATIONSHIP_PATH", f"No relationship path found for {field_name} in collection '{collection}' - applying restrictive filter")
                        rls_filters.append({"_id": {"$in": []}})  # This will match no documents
                except Exception as e:
                    logger.error(f"Error building relationship filter for {field_name} in {collection}: {e}")
                    # On error with all_involved, be restrictive
                    rls_filters.append({"_id": {"$in": []}})
            else:
                # base_only mode - skip fields that don't exist in collection
                self._audit_log("RLS_SKIP", f"Field '{field_name}' not found in collection '{collection}' and enforcement is base_only - skipping")
        
        # Combine RLS filters with base query
        if rls_filters:
            if len(rls_filters) == 1:
                rls_combined = rls_filters[0]
            else:
                rls_combined = {"$and": rls_filters}
            
            if enhanced_query:
                enhanced_query = {"$and": [enhanced_query, rls_combined]}
            else:
                enhanced_query = rls_combined
        else:
            # If no RLS filters applied and we have RLS values, log warning
            if rls_values:
                self._audit_log("RLS_WARNING", f"No RLS filters applied to collection '{collection}' - no suitable fields with values found")
        
        return enhanced_query
    
    def _build_relationship_filter(self, collection: str, field_name: str, field_value: str, reg: Dict) -> Optional[Dict]:
        """Build a MongoDB filter using relationship traversal for RLS enforcement.
        
        This is completely dynamic - works with ANY collection schema and relationship structure.
        No hardcoded healthcare/business logic.
        """
        try:
            # Get database connection
            cfg = get_connection_config(reg)
            if not cfg.get("mongo_uri") or not cfg.get("mongo_db"):
                return None
                
            client = MongoClient(cfg["mongo_uri"])
            db = client[cfg["mongo_db"]]
            
            # Dynamically find which collections have this field
            all_fields = get_all_fields(reg)
            collections_with_field = []
            
            for coll_name, fields in all_fields.items():
                field_names = [f["name"] for f in fields]
                if field_name in field_names:
                    collections_with_field.append(coll_name)
            
            if not collections_with_field:
                self._audit_log("RLS_WARNING", f"Field '{field_name}' not found in any collection")
                return None
            
            # Check if current collection already has the field
            current_fields = [f["name"] for f in all_fields.get(collection, [])]
            if field_name in current_fields:
                # Direct filtering - this shouldn't happen as we check this earlier
                return {field_name: field_value}
            
            # Dynamically find relationships from current collection to collections that have the field
            relations = get_collection_relations(collection, reg)
            
            # Method 1: Try direct relationships (works for ANY domain)
            for relation in relations:
                target_collection = relation.get("ref_collection")
                if target_collection in collections_with_field:
                    # This collection has a direct relationship to a collection with our field
                    local_field = relation.get("local_field")
                    foreign_field = relation.get("foreign_field")
                    
                    # Query the target collection to get matching foreign keys
                    target_docs = list(db[target_collection].find(
                        {field_name: field_value}, 
                        {foreign_field: 1, "_id": 0}
                    ))
                    
                    if target_docs:
                        matching_values = [doc[foreign_field] for doc in target_docs if foreign_field in doc]
                        if matching_values:
                            self._audit_log("RLS_RELATIONSHIP_FOUND", 
                                          f"Found {len(matching_values)} matching {local_field} values via {target_collection}")
                            return {local_field: {"$in": matching_values}}
                    
                    # If no matches found, return empty filter
                    return {local_field: {"$in": []}}
            
            # Method 2: Try reverse relationships (collections that point to us)
            all_collections = reg.get("collections", {})
            for other_collection, other_data in all_collections.items():
                if other_collection == collection:
                    continue
                    
                other_relations = other_data.get("relations", [])
                for relation in other_relations:
                    if relation.get("ref_collection") == collection:
                        # This collection points to us
                        if other_collection in collections_with_field:
                            # And it has our target field
                            other_local_field = relation.get("local_field")
                            our_foreign_field = relation.get("foreign_field")
                            
                            # Query the other collection
                            other_docs = list(db[other_collection].find(
                                {field_name: field_value}, 
                                {other_local_field: 1, "_id": 0}
                            ))
                            
                            if other_docs:
                                matching_values = [doc[other_local_field] for doc in other_docs if other_local_field in doc]
                                if matching_values:
                                    self._audit_log("RLS_REVERSE_RELATIONSHIP_FOUND", 
                                                  f"Found {len(matching_values)} matching {our_foreign_field} values via {other_collection}")
                                    return {our_foreign_field: {"$in": matching_values}}
                            
                            # If no matches found, return empty filter
                            return {our_foreign_field: {"$in": []}}
            
            # Method 3: Try two-hop relationships (through intermediary collections)
            for relation in relations:
                intermediary_collection = relation.get("ref_collection")
                local_field = relation.get("local_field")
                foreign_field = relation.get("foreign_field")
                
                # Check if intermediary has relationships to collections with our field
                intermediary_relations = get_collection_relations(intermediary_collection, reg)
                for inter_relation in intermediary_relations:
                    final_collection = inter_relation.get("ref_collection")
                    if final_collection in collections_with_field:
                        inter_local_field = inter_relation.get("local_field")
                        inter_foreign_field = inter_relation.get("foreign_field")
                        
                        # Query final collection first
                        final_docs = list(db[final_collection].find(
                            {field_name: field_value}, 
                            {inter_foreign_field: 1, "_id": 0}
                        ))
                        
                        if final_docs:
                            inter_values = [doc[inter_foreign_field] for doc in final_docs if inter_foreign_field in doc]
                            if inter_values:
                                # Query intermediary collection
                                inter_docs = list(db[intermediary_collection].find(
                                    {inter_local_field: {"$in": inter_values}}, 
                                    {foreign_field: 1, "_id": 0}
                                ))
                                
                                if inter_docs:
                                    final_values = [doc[foreign_field] for doc in inter_docs if foreign_field in doc]
                                    if final_values:
                                        self._audit_log("RLS_TWO_HOP_RELATIONSHIP_FOUND", 
                                                      f"Found {len(final_values)} matching {local_field} values via {intermediary_collection}->{final_collection}")
                                        return {local_field: {"$in": final_values}}
                        
                        # If no matches found, return empty filter
                        return {local_field: {"$in": []}}
            
            self._audit_log("RLS_NO_RELATIONSHIP_PATH", f"No relationship path found from {collection} to field {field_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error building relationship filter for {field_name} in {collection}: {e}")
            return None
    
    def _audit_log(self, action: str, message: str):
        """Log RLS actions for audit trail."""
        if self.config.get("audit_enabled", True):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user.user_id,
                "user_role": self.user.role,
                "action": action,
                "message": message
            }
            self.audit_log.append(log_entry)
            logger.info(f"RLS Audit: {action} - {message}")
    
    def get_audit_log(self) -> List[Dict]:
        """Get the audit log for this session."""
        return self.audit_log.copy()

# Factory function for easy integration
def create_rls_interceptor(user_id: str, role: str = "user", permissions: List[str] = None, team_id: str = None, session_id: str = None, patient_id: str = None) -> RLSQueryInterceptor:
    """Create an RLS interceptor for a user."""
    user_context = UserContext(
        user_id=user_id,
        role=role,
        permissions=permissions or [],
        team_id=team_id,
        session_id=session_id,
        patient_id=patient_id,
        bypass_rls=(role in ["admin", "super_user"])
    )
    
    return RLSQueryInterceptor(user_context)

# Example usage functions
def example_query_enhancement():
    """Example of how to use RLS query enhancement."""
    
    # Create user context
    user_context = UserContext(
        user_id="user123",
        role="customer",
        permissions=["read", "write"]
    )
    
    # Create RLS interceptor
    rls = RLSQueryInterceptor(user_context)
    
    # Example queries
    original_find = {"status": "active"}
    enhanced_find = rls.enhance_find_query("orders", original_find)
    print(f"Original: {original_find}")
    print(f"Enhanced: {enhanced_find}")
    
    # Example aggregation
    original_pipeline = [
        {"$match": {"category": "electronics"}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    enhanced_pipeline = rls.enhance_aggregate_pipeline("orders", original_pipeline)
    print(f"Original pipeline: {original_pipeline}")
    print(f"Enhanced pipeline: {enhanced_pipeline}")
    
    # Show audit log
    print("Audit log:", rls.get_audit_log())

if __name__ == "__main__":
    example_query_enhancement()
