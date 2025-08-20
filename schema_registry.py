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

