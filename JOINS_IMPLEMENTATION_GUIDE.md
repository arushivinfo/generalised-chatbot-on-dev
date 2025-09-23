# JOINS_IMPLEMENTATION_GUIDE.md

# Complete Joins Implementation Guide

This guide explains how the joins functionality has been implemented according to the objective.txt requirements.

## Overview

The joins implementation allows the Search Agent to handle queries that need data from multiple related collections while maintaining the existing JSON output format. The system automatically compiles and executes joins into valid MongoDB aggregation pipelines.

## Key Components Implemented

### 1. Search Agent Prompt Changes

**New JSON Output Format with Joins:**

```json
{
  "collection": "base_collection",
  "joins": [
    {
      "alias": "short_name",
      "collection": "target_collection", 
      "from": "base",
      "local_field": "field_in_base",
      "foreign_field": "field_in_target",
      "cardinality": "one_to_many",
      "join_type": "left"
    }
  ],
  "filters": [
    {"field": "base_field", "operation": "regex", "value": "test"},
    {"owner": "alias_name", "field": "joined_field", "operation": "regex", "value": "test"}
  ],
  "sort": {"field_name": "asc"},
  "limit": 20
}
```

**Key Features:**
- `joins[]` field only included when multiple collections are needed
- `owner` attribute in filters specifies which collection the filter applies to
- Base collection filters don't need `owner` attribute
- Joined field sorting uses dot notation: `"alias.field_name"`

### 2. Schema Registry Changes

**Relations Definition in Collections:**

Collections now include a `relations[]` section:

```json
{
  "collections": {
    "doctors": {
      "fields": [...],
      "relations": [
        {
          "alias": "appointments of doctors",
          "ref_collection": "appointments",
          "local_field": "doctor_id", 
          "foreign_field": "doctor_id",
          "cardinality": "many_to_one",
          "join_type": "inner"
        }
      ]
    }
  }
}
```

**New Functions Added:**
- `get_collection_relations(collection)` - Get relationships for a collection
- `set_collection_relations(collection, relations)` - Set relationships
- `validate_join_relationship(base_collection, join_spec)` - Validate joins against schema

### 3. Enhanced Pydantic Models

**New Models:**

```python
class EntityFilter(BaseModel):
    field: str
    operation: str
    value: Any
    owner: Optional[str] = None  # For joined fields

class Join(BaseModel):
    alias: str
    collection: str
    from_: str = "base"  # "base" or alias name
    local_field: str
    foreign_field: str
    cardinality: str = "one_to_many"
    join_type: str = "left"

class EntityQuery(BaseModel):
    filters: List[EntityFilter]
    joins: List[Join] = []  # Optional joins array
    sort: Dict[str, str] = {}
    limit: int = 20
```

### 4. MongoDB Executor Changes

**Dual Query Engine:**
- `_run_simple_query()` - For queries without joins (uses `find()`)
- `_run_aggregation_query()` - For queries with joins (uses aggregation pipeline)

**Aggregation Pipeline Generation:**

1. **Initial $match stage** - Base collection filters + RLS
2. **$lookup stages** - One per join definition
3. **$unwind stages** - For one_to_many relationships when needed
4. **Joined filters $match** - Filters on joined collections
5. **$sort stage** - Sorting on base and joined fields
6. **$limit stage** - Result limiting
7. **$project stage** - Remove _id field

### 5. Join Validation

**Schema Validation:**
- Validates joins against defined relationships in schema registry
- Checks field existence in target collections
- Warns about undefined relationships but doesn't block execution

**Runtime Validation:**
- Validates filter operations against field metadata
- Ensures proper field types and allowed operations
- Handles field type conversion for dates and numbers

### 6. RLS Integration with Joins

**Multi-layer RLS Support:**
- Base collection gets RLS filters in initial $match stage
- Joined collections can get RLS filters if `enforcement_mode = "all_involved"`
- Supports multiple authentication layers (user_id, department, team_id, etc.)

## Example Queries

### Example 1: Simple Base Query (No Joins)

**LLM Output:**
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

**Executor Action:** Uses `find()` with specialization filter and sorts by first_name.

### Example 2: Query with Join

**LLM Output:**
```json
{
  "collection": "doctors",
  "joins": [
    {
      "alias": "appointments",
      "collection": "appointments",
      "from": "base",
      "local_field": "doctor_id",
      "foreign_field": "doctor_id", 
      "cardinality": "one_to_many",
      "join_type": "left"
    }
  ],
  "filters": [
    {"field": "specialization", "operation": "regex", "value": "Pediatrics"},
    {"owner": "appointments", "field": "status", "operation": "regex", "value": "Scheduled"}
  ],
  "sort": {"first_name": "asc"},
  "limit": 10
}
```

**Generated MongoDB Pipeline:**
```javascript
[
  // Base filters + RLS
  {"$match": {"specialization": /Pediatrics/i, "user_id": "current_user"}},
  
  // Join appointments
  {"$lookup": {
    "from": "appointments",
    "localField": "doctor_id", 
    "foreignField": "doctor_id",
    "as": "appointments"
  }},
  
  // Unwind for filtering (one_to_many)
  {"$unwind": {"$path": "$appointments", "preserveNullAndEmptyArrays": true}},
  
  // Joined collection filters
  {"$match": {"appointments.status": /Scheduled/i}},
  
  // Sort and limit
  {"$sort": {"first_name": 1}},
  {"$limit": 10},
  {"$project": {"_id": 0}}
]
```

### Example 3: Complex Multi-Join Query

**LLM Output:**
```json
{
  "collection": "patients", 
  "joins": [
    {
      "alias": "appointments",
      "collection": "appointments",
      "from": "base", 
      "local_field": "patient_id",
      "foreign_field": "patient_id",
      "cardinality": "one_to_many",
      "join_type": "inner"
    },
    {
      "alias": "doctors", 
      "collection": "doctors",
      "from": "appointments",
      "local_field": "doctor_id",
      "foreign_field": "doctor_id", 
      "cardinality": "many_to_one",
      "join_type": "inner"
    }
  ],
  "filters": [
    {"field": "gender", "operation": "regex", "value": "F"},
    {"owner": "appointments", "field": "status", "operation": "regex", "value": "Completed"},
    {"owner": "doctors", "field": "specialization", "operation": "regex", "value": "Pediatrics"}
  ],
  "sort": {"appointments.appointment_date": "desc"},
  "limit": 5
}
```

## Admin UI Enhancements

### Relationship Editor

The Admin UI now includes a relationship editor for each collection:

1. **Add Relationship Button** - Create new relationships
2. **Relationship Form** - Configure alias, target collection, fields, cardinality, join type
3. **Validation** - Ensures proper relationship definition
4. **Visual Editor** - Drag-and-drop interface for complex relationships

### Schema Preview

The schema preview now shows available relationships for each collection, helping users understand available joins.

## Testing

Run the test suite to validate the implementation:

```bash
python test_joins_implementation.py
```

**Test Coverage:**
- Simple queries without joins
- Single collection with joins
- Multiple collection queries  
- RLS integration with joins
- Schema validation for joins
- Pydantic model parsing
- Complex join scenarios

## Performance Considerations

1. **Index Optimization** - Automatically creates indexes on join fields
2. **Pipeline Optimization** - Only adds $unwind when necessary for filtering/sorting
3. **Query Planning** - Validates joins early to fail fast on invalid relationships
4. **Memory Management** - Uses $limit early in pipeline when possible

## Security Features

1. **RLS Integration** - All joins respect Row-Level Security settings
2. **Schema Validation** - Prevents unauthorized collection access
3. **Field Validation** - Ensures only allowed operations on valid fields
4. **Audit Logging** - Tracks all join operations for security review

## Error Handling

1. **Join Validation Errors** - Clear messages for invalid relationships
2. **Field Validation Errors** - Specific errors for invalid fields/operations
3. **Pipeline Errors** - MongoDB aggregation error handling
4. **Graceful Degradation** - Falls back to simple queries when joins fail

## Migration Guide

**For Existing Queries:**
- All existing queries continue to work unchanged
- No breaking changes to existing functionality
- New joins functionality is additive only

**For Schema Updates:**
- Add `relations[]` section to existing collections
- Use Admin UI relationship editor for easy configuration
- Validate relationships after adding them

This implementation provides a complete, production-ready joins system that maintains backward compatibility while adding powerful new query capabilities.