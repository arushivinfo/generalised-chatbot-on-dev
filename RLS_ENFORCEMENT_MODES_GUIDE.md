# RLS Enforcement Modes: Complete Guide

## Overview

Your multi-layer RLS authentication system supports two enforcement modes that determine how filtering is applied when a field doesn't exist directly in the target collection.

## Enforcement Modes

### 1. **`base_only` Enforcement**

**What it does:**
- Only applies RLS filters to collections that **directly contain** the filter field
- If the target collection doesn't have the field, **no filtering is applied**
- Simple, fast, but less secure

**Example Scenario:**
```
Query: db.doctors.find({}) 
RLS Filter: patient_id = "P001"
Collection Field Check: doctors collection has no patient_id field
Result: No filtering applied → Returns ALL doctors
```

**When to use:**
- Performance is critical
- You only need basic isolation
- Collections are designed with direct ownership fields

### 2. **`all_involved` Enforcement**

**What it does:**
- Uses **relationship traversal** to apply filters across related collections
- When direct field doesn't exist, finds a path through relationships
- More secure but requires well-defined relationships

**Example Scenario:**
```
Query: db.doctors.find({})
RLS Filter: patient_id = "P001"
Collection Field Check: doctors collection has no patient_id field
Relationship Check: doctors ←→ appointments (via doctor_id)
Field Check: appointments collection HAS patient_id field
Process:
  1. Query: db.appointments.find({patient_id: "P001"}, {doctor_id: 1})
  2. Extract: ["D001", "D003", "D005"] 
  3. Apply: db.doctors.find({doctor_id: {$in: ["D001", "D003", "D005"]}})
Result: Only doctors associated with patient P001
```

**When to use:**
- Security is paramount
- Complex data relationships exist
- You need cross-collection filtering

## Real-World Hospital Example

### Your Schema Structure
```
patients: {patient_id, name, ...}
doctors: {doctor_id, name, specialization, ...}  ← No patient_id field
appointments: {appointment_id, patient_id, doctor_id, ...}  ← Bridge table
```

### Patient P001 Queries "Show my doctors"

#### With `base_only`:
```python
# Query: doctors collection
# Filter: patient_id = "P001"
# Check: doctors has patient_id field? → NO
# Action: Skip filtering
# Result: db.doctors.find({}) → ALL doctors returned ❌
```

#### With `all_involved`:
```python
# Query: doctors collection  
# Filter: patient_id = "P001"
# Check: doctors has patient_id field? → NO
# Relationship: doctors ←→ appointments (doctor_id)
# Check: appointments has patient_id field? → YES
# Traverse:
#   Step 1: db.appointments.find({patient_id: "P001"}, {doctor_id: 1})
#   Step 2: Extract doctor_ids: ["D001", "D003"] 
#   Step 3: db.doctors.find({doctor_id: {$in: ["D001", "D003"]}})
# Result: Only P001's doctors returned ✅
```

## Configuration Examples

### Global Setting
```json
{
  "rls_config": {
    "enforcement_mode": "all_involved",  // Global default
    "rls_layers": [...]
  }
}
```

### Per-Layer Override
```json
{
  "rls_layers": [
    {
      "field_name": "patient_id",
      "enforcement_mode": "all_involved",  // Uses traversal
      "enabled": true
    },
    {
      "field_name": "team_id", 
      "enforcement_mode": "base_only",     // Direct only
      "enabled": true
    }
  ]
}
```

### Mixed Configuration
```json
{
  "rls_config": {
    "enforcement_mode": "base_only",      // Global default
    "rls_layers": [
      {
        "field_name": "patient_id",
        "enforcement_mode": "all_involved"  // Override for this layer
      },
      {
        "field_name": "user_id",
        // Inherits global "base_only"
      }
    ]
  }
}
```

## Relationship Traversal Patterns

### 1. **Direct Relationship**
```
Collection A ←→ Collection B (via shared_field)
```

### 2. **Reverse Relationship**  
```
Collection B ←→ Collection A (via shared_field)
```

### 3. **Two-Hop Relationship**
```
Collection A ←→ Bridge ←→ Collection C
Example: doctors ←→ appointments ←→ treatments
```

## Performance Considerations

### `base_only`
- ✅ **Fast**: Single collection query
- ✅ **Simple**: No relationship lookups
- ❌ **Limited**: Only works with direct fields

### `all_involved`
- ✅ **Comprehensive**: Works across relationships
- ✅ **Secure**: Proper data isolation
- ❌ **Slower**: Multiple database queries
- ❌ **Complex**: Requires relationship definitions

## Best Practices

### 1. **Hierarchical Configuration**
```
Global default: "base_only" (performance)
Critical fields: "all_involved" (security)
```

### 2. **Field-Specific Strategy**
```
user_id: "base_only" (most collections have this)
patient_id: "all_involved" (needs traversal)
session_id: "base_only" (direct session isolation)
```

### 3. **Error Handling**
- `all_involved` mode: If no relationship path found → **restrictive filter** (empty results)
- `base_only` mode: If field not found → **no filtering** (all results)

## Testing Your Configuration

Run the test script to validate your enforcement modes:

```bash
python test_rls_enforcement_modes.py
```

This will show you exactly how your patient_id filtering behaves with both modes.

## Summary

**Use `base_only` when:**
- Performance is critical
- Collections have direct ownership fields
- Simple data model

**Use `all_involved` when:**
- Security requires cross-collection filtering  
- Complex relationships exist
- Data isolation is paramount

Your hospital system should use `all_involved` for `patient_id` filtering to ensure patients only see their own doctors, appointments, and treatments through proper relationship traversal.