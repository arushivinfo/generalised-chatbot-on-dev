# schema_registry.py
import json, os
from typing import Dict, List, Literal

REG_PATH = os.getenv("SCHEMA_REGISTRY_PATH", "schema_registry.json")

# role is used ONLY for core rules text; searching works for all collections
CoreRole = Literal["matches", "players", "venues", "upcoming_match", "other"]

DEFAULT = {
  "options_max": 20,  # default cap for categories shown in prompts (editable in Admin UI)
  "collections": {
    # seed with your four; you can add more from admin page
    "matches_filtered_90696": {
      "role": "matches", "description": "historical matches", "fields": []
    },
    "players_filtered_90696": {
      "role": "players", "description": "squad & match-wise player data", "fields": []
    },
    "venues_filtered_90696": {
      "role": "venues", "description": "venue & ground stats", "fields": []
    },
    "upcoming_match_90696_summary": {
      "role": "upcoming_match", "description": "upcoming match + prediction", "fields": []
    }
  },

  "core_rules": {
    "mode": "auto",          
    "custom_text": ""        
    },
  "user_match_context": ""   # optional admin-entered “extra add-up” context

}

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

def upsert_collection(coll_name: str, role: CoreRole, description: str, fields: List[Dict]) -> Dict:
    reg = load_registry()
    reg["collections"][coll_name] = {"role": role, "description": description, "fields": fields}
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

def get_core_coll_map(reg: Dict | None = None) -> Dict[str, str]:
    """
    Returns map for core rules: {'matches': <coll>, 'players': <coll>, 'venues': <coll>, 'upcoming_match': <coll>}
    Missing roles are simply omitted.
    """
    reg = reg or load_registry()
    out: Dict[str, str] = {}
    for coll, meta in reg["collections"].items():
        role = meta.get("role", "other")
        if role in ("matches", "players", "venues", "upcoming_match") and role not in out:
            out[role] = coll
    return out

def get_descriptions(reg: Dict | None = None) -> Dict[str, str]:
    reg = reg or load_registry()
    return {coll: meta.get("description","") for coll, meta in reg["collections"].items()}


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

