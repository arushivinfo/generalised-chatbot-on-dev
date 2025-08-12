_cache = []

def save_to_cache(query, answer):
    """Save the latest query and answer to in-memory cache."""
    global _cache
    print(f"Saving to cache: {query} -> {answer}")
    _cache.append({"query": query, "answer": answer})
    _cache = _cache[-20:]  # keep only last 20

def get_last_memories(n=1):
    """Get last n memories (query-answer pairs) from in-memory cache."""
    print(f"Loaded {len(_cache)} cached memories.") 
    print("Cache content:", _cache[-n:])
    return _cache[-n:] if _cache else []
def view_cache():
    """Return the full current cache (for debugging)."""
    return _cache