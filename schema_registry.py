# schema_registry.py
import json, os
from typing import Dict, List

REG_PATH = os.getenv("SCHEMA_REGISTRY_PATH", "schema_registry.json")

DEFAULT = {
  "options_max": 20,
  "collections": {
    # Start empty; populate via Admin → Collections
  },
  "core_rules": { "mode": "auto", "custom_text": "" },
  # Kept for backward compatibility with UI; label it “Extra Context”
  "user_match_context": ""
}

def get_connection_config(reg: dict | None = None):
    reg = reg or load_registry()
    return reg.get("connection", {})

def set_connection_config(uri: str, db: str):
    reg = load_registry()
    reg["connection"] = {"mongo_uri": uri, "mongo_db": db}
    save_registry(reg)
    return reg

def load_registry() -> Dict:  # {options_max:int, collections:{<coll_name>:{role,description,fields}}}
    if not os.path.exists(REG_PATH):
        save_registry(DEFAULT)
    with open(REG_PATH, "r") as f:
        return json.load(f)

def save_registry(reg: Dict) -> None:
    with open(REG_PATH, "w") as f:
        json.dump(reg, f, indent=2)

def list_collections() -> Dict[str, Dict]:
    return load_registry()["collections"]

def upsert_collection(coll_name: str, description: str, fields: List[Dict]) -> Dict:
    reg = load_registry()
    reg["collections"][coll_name] = {"description": description, "fields": fields}
    save_registry(reg); return reg

def delete_collection(coll_name: str) -> Dict:
    reg = load_registry()
    reg["collections"].pop(coll_name, None)
    save_registry(reg); return reg

def set_options_max(n: int) -> Dict:
    reg = load_registry(); reg["options_max"] = max(1, int(n)); save_registry(reg); return reg

def get_all_fields(reg: Dict | None = None) -> Dict[str, List[Dict]]:
    reg = reg or load_registry()
    return {coll: meta.get("fields", []) for coll, meta in reg["collections"].items()}



def get_descriptions(reg: Dict | None = None) -> Dict[str, str]:
    reg = reg or load_registry()
    return {coll: meta.get("description","") for coll, meta in reg["collections"].items()}

def get_collection_names(reg: Dict | None = None) -> List[str]:
    """List of collection names configured in Admin."""
    reg = reg or load_registry()
    return list(reg.get("collections", {}).keys())

def get_core_rules_config(reg: Dict | None = None) -> Dict:
    reg = reg or load_registry()
    cfg = reg.get("core_rules", {})
    return {"mode": cfg.get("mode", "auto"),
            "custom_text": cfg.get("custom_text", "")}

def set_core_rules_config(mode: str, custom_text: str) -> Dict:
    reg = load_registry()
    reg["core_rules"] = {"mode": mode, "custom_text": custom_text}
    save_registry(reg); return reg

def get_user_match_context(reg: Dict | None = None) -> str:
    reg = reg or load_registry()
    return reg.get("user_match_context", "")

def set_user_match_context(text: str) -> Dict:
    reg = load_registry()
    reg["user_match_context"] = text or ""
    save_registry(reg); return reg

# User access control functions
def get_user_access_config(reg: Dict | None = None) -> Dict:
    """Get the user access configuration from registry."""
    reg = reg or load_registry()
    if "user_access" not in reg:
        reg["user_access"] = {}  # Initialize if not exists
        save_registry(reg)
    return reg.get("user_access", {})

def set_user_access(user_id: str, collections: List[str]) -> Dict:
    """Set which collections a user can access."""
    reg = load_registry()
    if "user_access" not in reg:
        reg["user_access"] = {}
    reg["user_access"][user_id] = collections
    save_registry(reg)
    return reg

def delete_user_access(user_id: str) -> Dict:
    """Remove a user's access configuration."""
    reg = load_registry()
    if "user_access" in reg and user_id in reg["user_access"]:
        del reg["user_access"][user_id]
        save_registry(reg)
    return reg

def get_user_collections(user_id: str, reg: Dict | None = None) -> List[str]:
    """Get collections a user has access to. If user doesn't exist or has no
    specific permissions, return an empty list (no collections)."""
    reg = reg or load_registry()
    user_access = reg.get("user_access", {})
    # Return user's authorized collections or empty list if not found
    return user_access.get(user_id, [])

def get_user_accessible_collections(user_id: str, reg: Dict | None = None) -> List[str]:
    reg = reg or load_registry()
    if user_id == "admin":
        return get_collection_names(reg)
    user_access = reg.get("user_access", {})
    # if not user_access or user_id not in user_access:
    #     return get_collection_names(reg)
    return user_access.get(user_id, [])

def get_all_users(reg: Dict | None = None) -> List[str]:
    """Get list of all users with access configurations."""
    reg = reg or load_registry()
    return list(reg.get("user_access", {}).keys())

# Row-Level Security (RLS) functions
def get_rls_config(reg: Dict | None = None) -> Dict:
    """Get the Row-Level Security configuration."""
    reg = reg or load_registry()
    if "rls_config" not in reg:
        reg["rls_config"] = {
            "enabled": False,
            "rls_layers": [],  # New: multi-layer authentication
            "enforcement_mode": "base_only",  # "base_only" or "all_involved"
            "bypass_roles": ["admin", "super_user"],
            "audit_enabled": True,
            "collections": {}  # Per-collection overrides
        }
        save_registry(reg)
    return reg.get("rls_config", {})

def set_rls_config(config: Dict) -> Dict:
    """Set the Row-Level Security configuration."""
    reg = load_registry()
    reg["rls_config"] = config
    save_registry(reg)
    return reg

def set_rls_collection_field(collection: str, user_field: str, enforcement: str = None) -> Dict:
    """Set the user field for a specific collection."""
    reg = load_registry()
    rls_config = get_rls_config(reg)
    
    if "collections" not in rls_config:
        rls_config["collections"] = {}
    
    rls_config["collections"][collection] = {
        "user_field": user_field,
        "enforcement": enforcement or rls_config.get("enforcement_mode", "base_only")
    }
    
    reg["rls_config"] = rls_config
    save_registry(reg)
    return reg

def get_rls_collection_field(collection: str, reg: Dict | None = None) -> str:
    """Get the user field for a specific collection."""
    reg = reg or load_registry()
    rls_config = get_rls_config(reg)
    
    # Check collection-specific override
    if collection in rls_config.get("collections", {}):
        return rls_config["collections"][collection]["user_field"]
    
    # Return default
    return rls_config.get("default_user_field", "user_id")

def delete_rls_collection_config(collection: str) -> Dict:
    """Remove RLS configuration for a specific collection."""
    reg = load_registry()
    rls_config = get_rls_config(reg)
    
    if "collections" in rls_config and collection in rls_config["collections"]:
        del rls_config["collections"][collection]
        reg["rls_config"] = rls_config
        save_registry(reg)
    
    return reg

# Multi-layer RLS functions
def get_rls_layers(reg: Dict | None = None) -> List[Dict]:
    """Get the RLS authentication layers configuration."""
    reg = reg or load_registry()
    rls_config = get_rls_config(reg)
    return rls_config.get("rls_layers", [])

def set_rls_layers(layers: List[Dict]) -> Dict:
    """Set the RLS authentication layers configuration."""
    reg = load_registry()
    rls_config = get_rls_config(reg)
    rls_config["rls_layers"] = layers
    reg["rls_config"] = rls_config
    save_registry(reg)
    return reg

def get_available_rls_fields_for_collection(collection: str, reg: Dict | None = None) -> List[str]:
    """Get available RLS fields for a specific collection using enhanced validation."""
    reg = reg or load_registry()
    all_fields = get_all_fields(reg)
    
    if collection not in all_fields:
        return []
    
    # Use the enhanced validation function instead of pattern matching
    collection_fields = all_fields[collection]
    available_fields = []
    
    for field in collection_fields:
        if is_valid_rls_field(field):
            available_fields.append(field["name"])
    
    return available_fields

def is_valid_rls_field(field: Dict) -> bool:
    """Check if a field is suitable for RLS (Row-Level Security) filtering."""
    field_name = field.get("name", "").lower()
    field_type = field.get("type", "").lower()
    options = field.get("options", [])
    
    # Accept both string and numeric types for ID fields
    if field_type not in ["string", "int", "integer", "number"]:
        return False
    
    # Check if field name suggests it's an ID field (including healthcare patterns)
    id_patterns = [
        "user_id", "customer_id", "employee_id", "owner_id", "account_id", 
        "member_id", "patient_id", "doctor_id", "appointment_id", "treatment_id",
        "billing_id", "provider_id", "client_id", "department_id", "team_id",
        "organization_id", "company_id", "_id"
    ]
    
    # Check if field name contains any ID pattern
    if any(pattern in field_name for pattern in id_patterns):
        # For string fields, additional validation
        if field_type == "string":
            # If we have options, check if they look like IDs
            if options:
                # Check if options contain ID-like patterns (P001, U123, D001, etc.)
                id_like_options = [
                    opt for opt in options[:10] if isinstance(opt, str) and (
                        # Healthcare patterns: P001, D001, A001, etc.
                        (len(opt) >= 4 and opt[0].isalpha() and opt[1:].isdigit()) or
                        # Standard patterns: ID123, USER001, etc.
                        opt.upper().startswith(('P', 'U', 'D', 'C', 'E', 'A', 'B', 'ID', 'USER', 'CUST', 'DOC', 'PAT')) or 
                        # Numeric string IDs
                        opt.isdigit()
                    )
                ]
                return len(id_like_options) > 0
            else:
                # If no options but field name has ID pattern, it's probably valid
                return True
        
        # Numeric fields with id in name are always good
        if field_type in ["int", "integer", "number"]:
            return True
    
    return False

def get_available_rls_fields_for_collection_enhanced(collection: str, reg: Dict | None = None) -> List[str]:
    """Get available RLS fields for a specific collection with enhanced validation."""
    reg = reg or load_registry()
    all_fields = get_all_fields(reg)
    
    if collection not in all_fields:
        return []
    
    # Get fields that are suitable for RLS
    collection_fields = all_fields[collection]
    suitable_fields = []
    
    for field in collection_fields:
        if is_valid_rls_field(field):
            suitable_fields.append(field["name"])
    
    return suitable_fields

# Relationship/Join functions for collection schema
def get_collection_relations(collection: str, reg: Dict | None = None) -> List[Dict]:
    """Get relationship definitions for a collection."""
    reg = reg or load_registry()
    collections = reg.get("collections", {})
    if collection in collections:
        return collections[collection].get("relations", [])
    return []

def set_collection_relations(collection: str, relations: List[Dict]) -> Dict:
    """Set relationship definitions for a collection."""
    reg = load_registry()
    if "collections" not in reg:
        reg["collections"] = {}
    if collection not in reg["collections"]:
        reg["collections"][collection] = {"description": "", "fields": []}
    
    reg["collections"][collection]["relations"] = relations
    save_registry(reg)
    return reg

def validate_join_relationship(base_collection: str, join_spec: Dict, reg: Dict | None = None) -> bool:
    """Validate that a join relationship is defined in the schema."""
    reg = reg or load_registry()
    relations = get_collection_relations(base_collection, reg)
    
    target_collection = join_spec.get("collection")
    local_field = join_spec.get("local_field")
    foreign_field = join_spec.get("foreign_field")
    
    # Check if this exact relationship exists in schema
    for relation in relations:
        if (relation.get("ref_collection") == target_collection and
            relation.get("local_field") == local_field and
            relation.get("foreign_field") == foreign_field):
            return True
    
    # If not found, check reverse relationship (from target to base)
    target_relations = get_collection_relations(target_collection, reg)
    for relation in target_relations:
        if (relation.get("ref_collection") == base_collection and
            relation.get("local_field") == foreign_field and
            relation.get("foreign_field") == local_field):
            return True
    
    return False

def add_collection_relation(collection_name: str, relation: Dict) -> bool:
    """Add a relationship to a collection."""
    try:
        reg = load_registry()
        
        if collection_name not in reg.get("collections", {}):
            return False
        
        if "relations" not in reg["collections"][collection_name]:
            reg["collections"][collection_name]["relations"] = []
        
        # Validate required fields
        required_fields = ["alias", "ref_collection", "local_field", "foreign_field", "cardinality", "join_type"]
        if not all(field in relation for field in required_fields):
            return False
        
        # Check if relation with same alias already exists
        existing_relations = reg["collections"][collection_name]["relations"]
        if any(rel.get("alias") == relation.get("alias") for rel in existing_relations):
            return False  # Duplicate alias
        
        # Add the relation
        reg["collections"][collection_name]["relations"].append(relation)
        
        save_registry(reg)
        return True
    except Exception as e:
        print(f"Error adding relation: {e}")
        return False

def remove_collection_relation(collection_name: str, relation_index: int) -> bool:
    """Remove a relationship from a collection by index."""
    try:
        reg = load_registry()
        
        if collection_name not in reg.get("collections", {}):
            return False
        
        relations = reg["collections"][collection_name].get("relations", [])
        if 0 <= relation_index < len(relations):
            relations.pop(relation_index)
            reg["collections"][collection_name]["relations"] = relations
            save_registry(reg)
            return True
        
        return False
    except Exception as e:
        print(f"Error removing relation: {e}")
        return False

def get_all_collection_relationships() -> Dict[str, List[Dict]]:
    """Get all relationships across all collections."""
    reg = load_registry()
    relationships = {}
    
    for collection_name, data in reg.get("collections", {}).items():
        relations = data.get("relations", [])
        if relations:
            relationships[collection_name] = relations
    
    return relationships

def update_collection_relationship(collection_name: str, relation_index: int, updated_relation: Dict) -> bool:
    """Update an existing relationship in a collection."""
    try:
        reg = load_registry()
        
        if collection_name not in reg.get("collections", {}):
            return False
        
        relations = reg["collections"][collection_name].get("relations", [])
        if 0 <= relation_index < len(relations):
            # Validate required fields
            required_fields = ["alias", "ref_collection", "local_field", "foreign_field", "cardinality", "join_type"]
            if not all(field in updated_relation for field in required_fields):
                return False
            
            relations[relation_index] = updated_relation
            reg["collections"][collection_name]["relations"] = relations
            save_registry(reg)
            return True
        
        return False
    except Exception as e:
        print(f"Error updating relation: {e}")
        return False

def bulk_add_relationships(relationships_by_collection: Dict[str, List[Dict]]) -> Dict[str, int]:
    """
    Add multiple relationships across multiple collections.
    
    Args:
        relationships_by_collection: Dict mapping collection names to lists of relationships
        
    Returns:
        Dict with collection names and count of relationships added
    """
    result = {}
    reg = load_registry()
    
    for collection_name, relationships in relationships_by_collection.items():
        if collection_name not in reg.get("collections", {}):
            result[collection_name] = 0
            continue
        
        # Initialize relations if not exists
        if "relations" not in reg["collections"][collection_name]:
            reg["collections"][collection_name]["relations"] = []
        
        added_count = 0
        for relation in relationships:
            # Validate required fields
            required_fields = ["alias", "ref_collection", "local_field", "foreign_field", "cardinality", "join_type"]
            if all(field in relation for field in required_fields):
                reg["collections"][collection_name]["relations"].append(relation)
                added_count += 1
        
        result[collection_name] = added_count
    
    save_registry(reg)
    return result

def get_relationship_summary() -> Dict[str, any]:
    """Get a summary of all relationships in the system."""
    relationships = get_all_collection_relationships()
    
    total_relationships = sum(len(relations) for relations in relationships.values())
    collections_with_relations = len(relationships)
    
    # Get relationship patterns
    patterns = {}
    for collection, relations in relationships.items():
        for relation in relations:
            pattern = f"{collection} -> {relation['ref_collection']}"
            patterns[pattern] = patterns.get(pattern, 0) + 1
    
    return {
        "total_relationships": total_relationships,
        "collections_with_relations": collections_with_relations,
        "relationship_patterns": patterns,
        "collections": relationships
    }

def test_rls_field_validation():
    """Test RLS field validation for healthcare system fields."""
    
    # Test cases for healthcare system
    test_fields = [
        # Should be valid - string IDs with healthcare patterns
        {"name": "patient_id", "type": "string", "options": ["P001", "P002", "P003"]},
        {"name": "doctor_id", "type": "string", "options": ["D001", "D002", "D003"]}, 
        {"name": "user_id", "type": "string", "options": ["U001", "U002", "U003"]},
        
        # Should be valid - numeric IDs
        {"name": "user_id", "type": "int", "options": [1, 2, 3]},
        {"name": "customer_id", "type": "number", "options": [123, 456, 789]},
        
        # Should be invalid - wrong types
        {"name": "patient_name", "type": "string", "options": ["John", "Jane"]},
        {"name": "appointment_date", "type": "date", "options": []},
        {"name": "description", "type": "string", "options": []},
        
        # Edge cases
        {"name": "patient_id", "type": "string", "options": []},  # Should be valid - ID field even without options
        {"name": "some_field", "type": "string", "options": ["P001", "P002"]},  # Invalid - not an ID field name
    ]
    
    print("Testing RLS field validation:")
    for i, field in enumerate(test_fields):
        result = is_valid_rls_field(field)
        print(f"Test {i+1}: {field['name']} ({field['type']}) -> {'✅ Valid' if result else '❌ Invalid'}")
    
    print("\nTesting collection field detection:")
    # Test with actual registry data
    reg = load_registry()
    for collection_name in ["patients", "doctors", "appointments"]:
        if collection_name in reg.get("collections", {}):
            fields = reg["collections"][collection_name].get("fields", [])
            valid_fields = [f["name"] for f in fields if is_valid_rls_field(f)]
            print(f"{collection_name}: Valid RLS fields = {valid_fields}")

def debug_rls_field_detection():
    """Debug function to check what's happening with RLS field detection."""
    print("🔍 DEBUG: RLS Field Detection Analysis")
    print("=" * 50)
    
    reg = load_registry()
    all_fields = get_all_fields(reg)
    
    # Check each collection
    for collection_name, fields in all_fields.items():
        print(f"\n📄 Collection: {collection_name}")
        print(f"   Total fields: {len(fields)}")
        
        for field in fields:
            field_name = field.get("name", "MISSING_NAME")
            field_type = field.get("type", "MISSING_TYPE")
            options = field.get("options", [])
            
            # Debug the validation logic
            is_valid = is_valid_rls_field(field)
            
            print(f"   - {field_name} ({field_type}) -> {'✅' if is_valid else '❌'}")
            
            # Show detailed analysis for patient_id fields
            if "patient_id" in field_name.lower():
                print(f"     📋 Detailed analysis for {field_name}:")
                print(f"         Type check: {field_type in ['string', 'int', 'integer', 'number']}")
                print(f"         ID pattern match: {'patient_id' in field_name.lower()}")
                print(f"         Has options: {len(options) > 0} ({len(options)} options)")
                if options:
                    sample_options = options[:3]
                    print(f"         Sample options: {sample_options}")
                    # Check if options look like IDs
                    id_like = [
                        opt for opt in sample_options if isinstance(opt, str) and (
                            (len(opt) >= 4 and opt[0].isalpha() and opt[1:].isdigit()) or
                            opt.upper().startswith(('P', 'U', 'D', 'C', 'E', 'A', 'B', 'ID', 'USER', 'CUST', 'DOC', 'PAT')) or 
                            opt.isdigit()
                        )
                    ]
                    print(f"         ID-like options: {id_like}")
                    print(f"         ID-like count: {len(id_like)}")

if __name__ == "__main__":
    debug_rls_field_detection()

