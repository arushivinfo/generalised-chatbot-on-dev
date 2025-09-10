"""
Row-Level Security (RLS) Engine for MongoDB
Automatically filters queries based on user ownership and permissions.
"""

import re
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from schema_registry import get_rls_config, get_all_fields, load_registry
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    """User context for RLS filtering."""
    user_id: str
    role: str
    permissions: List[str]
    bypass_rls: bool = False

class RLSFieldDetector:
    """AI and heuristic-based field detection for RLS."""
    
    OWNERSHIP_PATTERNS = {
        "exact_matches": ["user_id", "customer_id", "employee_id", "owner_id", "account_id", "member_id"],
        "suffix_patterns": ["_user_id", "_customer_id", "_owner_id", "_account_id", "_member_id"],
        "contextual_patterns": ["user.*_id", "customer.*_id", "owner.*_id", "account.*_id", "member.*_id"]
    }
    
    def __init__(self):
        self.llm = None
        try:
            from admin_schema_ui import get_llm
            self.llm = get_llm()
        except Exception as e:
            logger.warning(f"AI field detection unavailable: {e}")
    
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
        
        # 4. Try AI detection if available
        if self.llm:
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
        if not self.llm:
            return None
        
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
            response = self.llm.invoke(prompt)
            suggested_field = response.content.strip().strip('"').strip("'")
            
            # Validate the suggestion
            field_names = [f["name"] for f in fields]
            if suggested_field in field_names and suggested_field != "null":
                return suggested_field
            
        except Exception as e:
            logger.error(f"AI field detection error: {e}")
        
        return None

class RLSQueryInterceptor:
    """MongoDB query interceptor for Row-Level Security."""
    
    def __init__(self, user_context: UserContext, rls_config: Dict = None):
        self.user = user_context
        self.config = rls_config or get_rls_config()
        self.field_detector = RLSFieldDetector()
        self.audit_log = []
    
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
        """Enhance a find query with RLS filter."""
        if self.should_bypass_rls():
            return query
        
        ownership_field = self.resolve_ownership_field(collection)
        if not ownership_field:
            self._audit_log("RLS_WARNING", f"No ownership field found for collection '{collection}'")
            return query

        # Detect field type from schema
        reg = load_registry()
        all_fields = get_all_fields(reg)
        field_type = None
        for f in all_fields.get(collection, []):
            if f["name"] == ownership_field:
                field_type = f.get("type")
                break

        user_id_val = self.user.user_id
        if field_type == "int":
            try:
                user_id_val = int(user_id_val)
            except Exception:
                pass

        enhanced_query = query.copy()
        enhanced_query[ownership_field] = user_id_val

        self._audit_log("RLS_FILTER", f"Added filter {ownership_field}={user_id_val} to {collection}")
        return enhanced_query
    
    def enhance_find_query_with_multilayer(self, collection: str, query: Dict, rls_values: Dict = None) -> Dict:
        """Enhance a find query with multi-layer RLS filtering."""
        if self.should_bypass_rls():
            return query
        
        # Get configured RLS layers
        rls_layers = self.config.get("rls_layers", [])
        if not rls_layers and not rls_values:
            # Fall back to single-layer RLS
            return self.enhance_find_query(collection, query)
        
        # Check which fields are available in this collection
        reg = load_registry()
        all_fields = get_all_fields(reg)
        collection_fields = [f["name"] for f in all_fields.get(collection, [])]
        
        enhanced_query = query.copy()
        rls_filters = []
        
        # Apply each RLS layer if the field exists in this collection
        for layer in rls_layers:
            if not layer.get("enabled", True):
                continue
                
            field_name = layer.get("field_name")
            if not field_name:
                continue
                
            # Only apply filter if field exists in collection
            if field_name not in collection_fields:
                self._audit_log("RLS_SKIP", f"Field '{field_name}' not found in collection '{collection}' - skipping layer")
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
                
            if field_value is not None and str(field_value).strip():
                # Detect field type from schema
                field_type = None
                for f in all_fields.get(collection, []):
                    if f["name"] == field_name:
                        field_type = f.get("type")
                        break
                
                # Convert value to appropriate type
                if field_type == "int":
                    try:
                        field_value = int(field_value)
                    except Exception:
                        pass
                
                rls_filters.append({field_name: field_value})
                self._audit_log("RLS_LAYER_APPLIED", f"Applied RLS layer {field_name}={field_value} to {collection}")
        
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
        
        return enhanced_query
    
    def enhance_aggregate_pipeline(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        """Enhance an aggregation pipeline with RLS filters."""
        if self.should_bypass_rls():
            return pipeline
        
        ownership_field = self.resolve_ownership_field(collection)
        if not ownership_field:
            return pipeline
        
        # Add RLS filter as first stage if no $match exists, or enhance existing $match
        enhanced_pipeline = pipeline.copy()
        
        # Find first $match stage or insert at beginning
        rls_filter = {ownership_field: self.user.user_id}
        
        if enhanced_pipeline and "$match" in enhanced_pipeline[0]:
            # Enhance existing $match
            enhanced_pipeline[0]["$match"].update(rls_filter)
        else:
            # Insert new $match at beginning
            enhanced_pipeline.insert(0, {"$match": rls_filter})
        
        # Handle enforcement mode for $lookup stages
        enforcement = self.config.get("enforcement_mode", "base_only")
        if enforcement == "all_involved":
            enhanced_pipeline = self._enhance_lookups_with_rls(enhanced_pipeline)
        
        self._audit_log("RLS_PIPELINE", f"Enhanced aggregation pipeline for {collection}")
        return enhanced_pipeline
    
    def enhance_update_query(self, collection: str, filter_query: Dict, update_doc: Dict) -> Tuple[Dict, Dict]:
        """Enhance an update query with RLS filter."""
        if self.should_bypass_rls():
            return filter_query, update_doc
        
        ownership_field = self.resolve_ownership_field(collection)
        if not ownership_field:
            return filter_query, update_doc
        
        enhanced_filter = filter_query.copy()
        enhanced_filter[ownership_field] = self.user.user_id
        
        self._audit_log("RLS_UPDATE", f"Added RLS filter to update on {collection}")
        return enhanced_filter, update_doc
    
    def enhance_delete_query(self, collection: str, query: Dict) -> Dict:
        """Enhance a delete query with RLS filter."""
        if self.should_bypass_rls():
            return query
        
        ownership_field = self.resolve_ownership_field(collection)
        if not ownership_field:
            return query
        
        enhanced_query = query.copy()
        enhanced_query[ownership_field] = self.user.user_id
        
        self._audit_log("RLS_DELETE", f"Added RLS filter to delete on {collection}")
        return enhanced_query
    
    def _enhance_lookups_with_rls(self, pipeline: List[Dict]) -> List[Dict]:
        """Add RLS filters to $lookup stages when enforcement_mode is 'all_involved'."""
        enhanced = []
        
        for stage in pipeline:
            if "$lookup" in stage:
                lookup = stage["$lookup"]
                foreign_collection = lookup.get("from")
                
                if foreign_collection:
                    foreign_field = self.resolve_ownership_field(foreign_collection)
                    if foreign_field:
                        # Add RLS filter to lookup pipeline
                        if "pipeline" not in lookup:
                            lookup["pipeline"] = []
                        
                        # Insert RLS filter at beginning of lookup pipeline
                        rls_match = {"$match": {foreign_field: self.user.user_id}}
                        lookup["pipeline"].insert(0, rls_match)
            
            enhanced.append(stage)
        
        return enhanced
    
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
def create_rls_interceptor(user_id: str, role: str = "user", permissions: List[str] = None) -> RLSQueryInterceptor:
    """Create an RLS interceptor for a user."""
    user_context = UserContext(
        user_id=user_id,
        role=role,
        permissions=permissions or [],
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
