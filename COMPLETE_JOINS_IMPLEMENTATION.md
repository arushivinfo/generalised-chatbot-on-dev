# Complete Joins Implementation

## Overview

This implementation provides complete joins functionality for the Search Agent according to the objective.txt requirements. The system now supports:

1. **Relationship-aware queries** - Automatically joins related collections
2. **Schema-based validation** - Validates joins against defined relationships  
3. **Intelligent query planning** - Uses joins when data spans multiple collections
4. **RLS integration** - Row-level security works with joined queries

## Key Components

### 1. Enhanced Schema Registry (`schema_registry.py`)

**New Functions:**
- `get_collection_relations(collection)` - Get relationships for a collection
- `validate_join_relationship(base, join_spec)` - Validate joins against schema
- `add_collection_relation()` / `remove_collection_relation()` - Manage relationships

**Relationship Schema:**
```json
{
  "collections": {
    "patients": {
      "relations": [
        {
          "alias": "appointments of patients",
          "ref_collection": "appointments",
          "local_field": "patient_id",
          "foreign_field": "patient_id", 
          "cardinality": "one_to_many",
          "join_type": "left"
        }
      ]
    }
  }
}
```

### 2. Enhanced Core Rules (`core_rules.py`)

**New Function:**
- `render_schema_section_with_relations()` - Includes relationship info in schema prompts

**Output Format:**
```
• patients [Patient information] → patient_id (string) [regex,sort]; first_name (string) [regex,sort]...
  Available Joins: JOIN appointments→appointments ON patient_id=patient_id (one_to_many,left)
```

### 3. Updated Search Agent (`search_agent_new.py`)

**Enhanced Features:**
- **Dual Query Engine**: `_run_simple_query()` for single collection, `_run_aggregation_query()` for joins
- **Smart Join Detection**: Automatically detects when joins are needed
- **MongoDB Pipeline Generation**: Converts joins to `$lookup`, `$unwind`, `$match` stages
- **RLS Integration**: Row-level security works with aggregation pipelines

**New Pydantic Models:**
```python
class EntityFilter(BaseModel):
    field: str
    operation: str
    value: Any
    owner: Optional[str] = None  # For joined fields

class Join(BaseModel):
    alias: str
    collection: str
    from_: str = "base"
    local_field: str
    foreign_field: str
    cardinality: str = "one_to_many"
    join_type: str = "left"

class EntityQuery(BaseModel):
    filters: List[EntityFilter]
    joins: List[Join] = []  # Optional joins
    sort: Dict[str, str] = {}
    limit: int = 20
```

### 4. Enhanced System Prompt

**Key Improvements:**
- **Clear join decision rules** - When to use joins vs single collection
- **Wrong vs Right examples** - Shows incorrect approaches and correct alternatives  
- **Relationship awareness** - Uses available joins from schema
- **Filter ownership** - Uses `owner` attribute for joined field filters

## Usage Examples

### Simple Query (No Joins)
```json
{
  "collection": "doctors",
  "filters": [
    {"field": "specialization", "operation": "regex", "value": "Pediatrics"}
  ],
  "sort": {"first_name": "asc"},
  "limit": 10
}
```

### Query with Joins
```json
{
  "collection": "patients",
  "joins": [
    {
      "alias": "appt",
      "collection": "appointments",
      "from": "base", 
      "local_field": "patient_id",
      "foreign_field": "patient_id",
      "cardinality": "one_to_many",
      "join_type": "left"
    }
  ],
  "filters": [
    {"field": "first_name", "operation": "regex", "value": "david"},
    {"owner": "appt", "field": "status", "operation": "regex", "value": "Scheduled"}
  ],
  "sort": {"appt.appointment_date": "asc"},
  "limit": 20
}
```

### Generated MongoDB Pipeline
```javascript
[
  {"$match": {"first_name": /david/i, "user_id": "current_user"}},
  {"$lookup": {
    "from": "appointments",
    "localField": "patient_id",
    "foreignField": "patient_id", 
    "as": "appt"
  }},
  {"$unwind": {"path": "$appt", "preserveNullAndEmptyArrays": true}},
  {"$match": {"appt.status": /Scheduled/i}},
  {"$sort": {"appt.appointment_date": 1}},
  {"$limit": 20},
  {"$project": {"_id": 0}}
]
```

## Real-World Query Examples

### ❌ Before (Wrong Approach)
**Query**: "Show appointments for patient David"
```json
{
  "collection": "appointments",
  "filters": [
    {"field": "patient_id", "operation": "regex", "value": "david"}
  ]
}
```
**Problem**: Searches for "david" in patient_id field instead of patient name.

### ✅ After (Correct Approach)  
```json
{
  "collection": "patients",
  "joins": [
    {
      "alias": "appt",
      "collection": "appointments",
      "from": "base",
      "local_field": "patient_id", 
      "foreign_field": "patient_id",
      "cardinality": "one_to_many",
      "join_type": "left"
    }
  ],
  "filters": [
    {"field": "first_name", "operation": "regex", "value": "david"}
  ],
  "sort": {"appt.appointment_date": "asc"}
}
```
**Result**: Correctly joins patients and appointments to find David's appointments.

## Testing

Run the comprehensive test suite:
```bash
python test_complete_joins.py
```

**Test Coverage:**
- ✅ Schema with relationships rendering
- ✅ Join validation against schema
- ✅ Pydantic model validation
- ✅ Search agent join generation
- ✅ MongoDB pipeline execution

## Migration Guide

### For Existing Queries
- **No breaking changes** - All existing single-collection queries work unchanged
- **Automatic enhancement** - Queries that need joins will be auto-upgraded

### For Schema Updates
1. Add `relations[]` section to collections that have relationships
2. Use the admin UI relationship editor for easy configuration
3. Validate relationships after adding them

### Example Relationship Addition
```python
from schema_registry import add_collection_relation

# Add relationship: patients → appointments
relation = {
    "alias": "patient appointments",
    "ref_collection": "appointments", 
    "local_field": "patient_id",
    "foreign_field": "patient_id",
    "cardinality": "one_to_many",
    "join_type": "left"
}

add_collection_relation("patients", relation)
```

## Performance Considerations

1. **Automatic Indexing** - Creates indexes on join fields automatically
2. **Pipeline Optimization** - Only adds `$unwind` when necessary for filtering/sorting
3. **Early Filtering** - Applies base collection filters before joins
4. **Limit Optimization** - Uses `$limit` early in pipeline when possible

## Security Features

1. **RLS Integration** - All joins respect Row-Level Security settings
2. **Schema Validation** - Prevents unauthorized collection access via joins
3. **Field Validation** - Ensures only allowed operations on valid fields
4. **Audit Logging** - Tracks all join operations for security review

## Future Enhancements

1. **Multi-level Joins** - Support for joins between more than 2 collections
2. **Query Optimization** - Advanced MongoDB pipeline optimization
3. **Visual Query Builder** - Drag-and-drop interface for complex joins
4. **Performance Monitoring** - Query execution time tracking and optimization suggestions

---

This implementation provides a complete, production-ready joins system that maintains backward compatibility while adding powerful new query capabilities.