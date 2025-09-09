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
            "default_user_field": "user_id",
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

