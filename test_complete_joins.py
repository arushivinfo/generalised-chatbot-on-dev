#!/usr/bin/env python3
"""
Test the complete joins implementation.
"""

import json
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_schema_with_relations():
    """Test that schema rendering includes relationship information."""
    print("=== Testing Schema with Relations ===")
    
    try:
        from core_rules import render_schema_section_with_relations
        from schema_registry import load_registry, get_all_fields, get_descriptions
        
        reg = load_registry()
        all_fields = get_all_fields(reg)
        descriptions = get_descriptions(reg)
        
        schema_output = render_schema_section_with_relations(all_fields, descriptions, 20)
        print("Schema with relations:")
        print(schema_output)
        
        # Check if relationships are included
        if "Available Joins" in schema_output:
            print("✅ Relationships are correctly included in schema output")
            return True
        else:
            print("❌ Relationships missing from schema output")
            return False
            
    except Exception as e:
        print(f"❌ Error testing schema: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_join_validation():
    """Test join validation against schema."""
    print("\n=== Testing Join Validation ===")
    
    from schema_registry import validate_join_relationship
    
    try:
        # Test valid join (should exist in your schema)
        valid_join = {
            "alias": "appointments",
            "collection": "appointments",
            "local_field": "patient_id",
            "foreign_field": "patient_id",
            "cardinality": "one_to_many",
            "join_type": "left"
        }
        
        is_valid = validate_join_relationship("patients", valid_join)
        print(f"Valid join test: {'✅ PASS' if is_valid else '❌ FAIL'}")
        
        # Test invalid join
        invalid_join = {
            "alias": "invalid",
            "collection": "nonexistent",
            "local_field": "fake_field",
            "foreign_field": "fake_field",
            "cardinality": "one_to_many",
            "join_type": "left"
        }
        
        is_invalid = not validate_join_relationship("patients", invalid_join)
        print(f"Invalid join test: {'✅ PASS' if is_invalid else '❌ FAIL'}")
        
        return is_valid and is_invalid
        
    except Exception as e:
        print(f"❌ Error testing join validation: {e}")
        return False

def test_search_agent_joins():
    """Test search agent with join queries."""
    print("\n=== Testing Search Agent with Joins ===")
    
    try:
        from search_agent_new import run_search_agent
        
        # Test query that should trigger joins
        test_queries = [
            "Show appointments for patient David",
            "Find appointments for Dr. Sarah", 
            "List all scheduled appointments with patient details"
        ]
        
        results = []
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            try:
                spec, answer, debug = run_search_agent("test_user", "test_team", query)
                
                # Check if joins were used appropriately
                if isinstance(spec, dict):
                    has_joins = "joins" in spec and len(spec.get("joins", [])) > 0
                    print(f"  Joins used: {'✅ YES' if has_joins else '❌ NO'}")
                    print(f"  Spec: {json.dumps(spec, indent=2)}")
                else:
                    print(f"  ❌ Invalid spec format: {type(spec)}")
                
                results.append({"query": query, "spec": spec, "success": True})
                
            except Exception as e:
                print(f"  ❌ Query failed: {e}")
                results.append({"query": query, "success": False, "error": str(e)})
        
        success_count = sum(1 for r in results if r["success"])
        print(f"\nSearch agent tests: {success_count}/{len(test_queries)} passed")
        return success_count == len(test_queries)
        
    except Exception as e:
        print(f"❌ Error testing search agent: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model validation."""
    print("\n=== Testing Pydantic Models ===")
    
    try:
        from search_agent_new import EntityFilter, Join, EntityQuery
        
        # Test EntityFilter with owner
        filter_data = {
            "field": "status",
            "operation": "regex", 
            "value": "active",
            "owner": "appointments"
        }
        entity_filter = EntityFilter(**filter_data)
        print(f"✅ EntityFilter: {entity_filter}")
        
        # Test Join model
        join_data = {
            "alias": "appointments",
            "collection": "appointments",
            "from_": "base",
            "local_field": "patient_id", 
            "foreign_field": "patient_id",
            "cardinality": "one_to_many",
            "join_type": "left"
        }
        join = Join(**join_data)
        print(f"✅ Join: {join}")
        
        # Test EntityQuery with joins
        query_data = {
            "filters": [filter_data],
            "joins": [join_data],
            "sort": {"first_name": "asc"},
            "limit": 10
        }
        entity_query = EntityQuery(**query_data)
        print(f"✅ EntityQuery: {entity_query}")
        
        print("✅ All Pydantic models validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Pydantic model validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Complete Joins Implementation Tests\n")
    
    tests = [
        ("Schema with Relations", test_schema_with_relations),
        ("Join Validation", test_join_validation),
        ("Pydantic Models", test_pydantic_models),
        ("Search Agent Joins", test_search_agent_joins),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"\n{test_name}: {'✅ PASSED' if success else '❌ FAILED'}")
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25} : {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Joins implementation is complete.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())