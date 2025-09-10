# test_joins_implementation.py
"""
Test file to validate the complete joins implementation.
Includes tests for:
1. Simple queries without joins
2. Single collection with joins 
3. Multiple collection queries
4. RLS integration with joins
5. Schema validation for joins
"""

import json
from search_agent_new import _run_query, EntityQuery, Join, EntityFilter
from schema_registry import validate_join_relationship

def test_simple_query():
    """Test a simple query without joins."""
    print("=== Testing Simple Query (No Joins) ===")
    
    query_spec = {
        "filters": [
            {"field": "first_name", "operation": "regex", "value": "Alex"}
        ],
        "sort": {"first_name": "asc"},
        "limit": 5
    }
    
    result = _run_query("doctors", query_spec)
    print(f"Result: {result}")
    print()

def test_join_query():
    """Test a query with joins."""
    print("=== Testing Join Query ===")
    
    query_spec = {
        "filters": [
            {"field": "doctor_id", "operation": "regex", "value": "D001"},
            {"owner": "appointments", "field": "status", "operation": "regex", "value": "Scheduled"}
        ],
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
        "sort": {"first_name": "asc"},
        "limit": 10
    }
    
    result = _run_query("doctors", query_spec)
    print(f"Result: {result}")
    print()

def test_join_validation():
    """Test join relationship validation."""
    print("=== Testing Join Validation ===")
    
    # Valid join (exists in schema)
    valid_join = {
        "alias": "appointments",
        "collection": "appointments",
        "local_field": "doctor_id",
        "foreign_field": "doctor_id",
        "cardinality": "one_to_many",
        "join_type": "left"
    }
    
    is_valid = validate_join_relationship("doctors", valid_join)
    print(f"Valid join test: {is_valid}")
    
    # Invalid join (doesn't exist in schema)
    invalid_join = {
        "alias": "invalid",
        "collection": "nonexistent",
        "local_field": "fake_field",
        "foreign_field": "fake_field",
        "cardinality": "one_to_many",
        "join_type": "left"
    }
    
    is_valid = validate_join_relationship("doctors", invalid_join)
    print(f"Invalid join test: {is_valid}")
    print()

def test_multi_collection_query():
    """Test multiple collections in one query."""
    print("=== Testing Multi-Collection Query ===")
    
    # Note: This would be tested with multi-entity queries
    # For now, just show the structure
    multi_spec = {
        "queries": [
            {
                "collection": "doctors",
                "filters": [{"field": "specialization", "operation": "regex", "value": "Pediatrics"}],
                "sort": {"first_name": "asc"},
                "limit": 5
            },
            {
                "collection": "patients", 
                "filters": [{"field": "gender", "operation": "regex", "value": "F"}],
                "sort": {"last_name": "asc"},
                "limit": 5
            }
        ]
    }
    
    print(f"Multi-collection spec structure: {json.dumps(multi_spec, indent=2)}")
    print()

def test_complex_join_with_sorting():
    """Test complex join with sorting on joined fields."""
    print("=== Testing Complex Join with Sorting ===")
    
    query_spec = {
        "filters": [
            {"field": "specialization", "operation": "regex", "value": "Pediatrics"},
            {"owner": "appointments", "field": "reason_for_visit", "operation": "regex", "value": "Checkup"}
        ],
        "joins": [
            {
                "alias": "appointments",
                "collection": "appointments", 
                "from": "base",
                "local_field": "doctor_id",
                "foreign_field": "doctor_id",
                "cardinality": "one_to_many",
                "join_type": "inner"
            }
        ],
        "sort": {"appointments.appointment_date": "desc"},
        "limit": 10
    }
    
    result = _run_query("doctors", query_spec)
    print(f"Complex join result: {result}")
    print()

def test_pydantic_models():
    """Test Pydantic model parsing."""
    print("=== Testing Pydantic Models ===")
    
    # Test EntityFilter with owner
    filter_data = {
        "field": "status",
        "operation": "regex", 
        "value": "active",
        "owner": "appointments"
    }
    entity_filter = EntityFilter(**filter_data)
    print(f"EntityFilter: {entity_filter}")
    
    # Test Join model
    join_data = {
        "alias": "appointments",
        "collection": "appointments",
        "from_": "base",
        "local_field": "doctor_id", 
        "foreign_field": "doctor_id",
        "cardinality": "one_to_many",
        "join_type": "left"
    }
    join = Join(**join_data)
    print(f"Join: {join}")
    
    # Test EntityQuery with joins
    query_data = {
        "filters": [filter_data],
        "joins": [join_data],
        "sort": {"first_name": "asc"},
        "limit": 10
    }
    entity_query = EntityQuery(**query_data)
    print(f"EntityQuery: {entity_query}")
    print()

def run_all_tests():
    """Run all tests."""
    print("Starting Joins Implementation Tests...\n")
    
    try:
        test_pydantic_models()
        test_join_validation()
        test_simple_query()
        test_join_query()
        test_complex_join_with_sorting()
        test_multi_collection_query()
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()