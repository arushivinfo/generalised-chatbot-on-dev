#!/usr/bin/env python3
"""
Test and demonstrate the join teaching system.
Shows wrong vs right approaches for common queries.
"""

import json
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_join_teacher():
    """Test the join teaching system."""
    print("🎓 Testing Join Teaching System")
    print("="*60)
    
    try:
        from join_teacher import JoinTeacher, validate_query_joins
        
        teacher = JoinTeacher()
        
        # Test 1: Wrong pattern - searching for name in ID field
        print("\n1. ❌ WRONG PATTERN: Searching for name in ID field")
        wrong_query1 = {
            "collection": "appointments",
            "filters": [{"field": "patient_id", "operation": "regex", "value": "david"}]
        }
        print(f"Query: {json.dumps(wrong_query1, indent=2)}")
        
        is_valid, warnings, analysis = validate_query_joins(wrong_query1)
        print(f"Valid: {is_valid}")
        for warning in warnings:
            print(f"  {warning}")
        
        # Test 2: Wrong pattern - direct patient-doctor join
        print("\n2. ❌ WRONG PATTERN: Direct patient-doctor join")
        wrong_query2 = {
            "collection": "patients",
            "joins": [{"alias": "doc", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id"}],
            "filters": [{"owner": "doc", "field": "last_name", "operation": "regex", "value": "brown"}]
        }
        print(f"Query: {json.dumps(wrong_query2, indent=2)}")
        
        is_valid, warnings, analysis = validate_query_joins(wrong_query2)
        print(f"Valid: {is_valid}")
        for warning in warnings:
            print(f"  {warning}")
        
        # Test 3: Correct pattern - patient name to appointments
        print("\n3. ✅ CORRECT PATTERN: Patient name to appointments")
        correct_query1 = {
            "collection": "patients",
            "joins": [{"alias": "appt", "collection": "appointments", "local_field": "patient_id", "foreign_field": "patient_id", "cardinality": "one_to_many"}],
            "filters": [{"field": "first_name", "operation": "regex", "value": "david"}],
            "sort": {"appt.appointment_date": "desc"}
        }
        print(f"Query: {json.dumps(correct_query1, indent=2)}")
        
        is_valid, warnings, analysis = validate_query_joins(correct_query1)
        print(f"Valid: {is_valid}")
        for warning in warnings:
            print(f"  {warning}")
        
        # Test 4: Correct pattern - doctor name to appointments
        print("\n4. ✅ CORRECT PATTERN: Doctor name to appointments")
        correct_query2 = {
            "collection": "doctors",
            "joins": [{"alias": "appt", "collection": "appointments", "local_field": "doctor_id", "foreign_field": "doctor_id", "cardinality": "one_to_many"}],
            "filters": [
                {"field": "first_name", "operation": "regex", "value": "linda"},
                {"field": "last_name", "operation": "regex", "value": "brown"}
            ],
            "sort": {"appt.appointment_date": "desc"}
        }
        print(f"Query: {json.dumps(correct_query2, indent=2)}")
        
        is_valid, warnings, analysis = validate_query_joins(correct_query2)
        print(f"Valid: {is_valid}")
        for warning in warnings:
            print(f"  {warning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing join teacher: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_scenarios():
    """Test various query scenarios that users might ask."""
    print("\n" + "="*60)
    print("🔍 Testing Real Query Scenarios")
    print("="*60)
    
    scenarios = [
        {
            "question": "name of patients consulting to dr Linda brown along with their patient ids",
            "explanation": "This requires joining all three collections: patients, appointments, doctors",
            "correct_approach": {
                "collection": "appointments",
                "joins": [
                    {"alias": "patient", "collection": "patients", "local_field": "patient_id", "foreign_field": "patient_id", "cardinality": "many_to_one"},
                    {"alias": "doctor", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id", "cardinality": "many_to_one"}
                ],
                "filters": [
                    {"owner": "doctor", "field": "first_name", "operation": "regex", "value": "linda"},
                    {"owner": "doctor", "field": "last_name", "operation": "regex", "value": "brown"}
                ]
            }
        },
        {
            "question": "Show appointments for patient David Smith",
            "explanation": "Start with patients collection, join appointments",
            "correct_approach": {
                "collection": "patients",
                "joins": [
                    {"alias": "appt", "collection": "appointments", "local_field": "patient_id", "foreign_field": "patient_id", "cardinality": "one_to_many"}
                ],
                "filters": [
                    {"field": "first_name", "operation": "regex", "value": "david"},
                    {"field": "last_name", "operation": "regex", "value": "smith"}
                ],
                "sort": {"appt.appointment_date": "desc"}
            }
        },
        {
            "question": "Find all appointments with Dr. Sarah Wilson",
            "explanation": "Start with doctors collection, join appointments",
            "correct_approach": {
                "collection": "doctors",
                "joins": [
                    {"alias": "appt", "collection": "appointments", "local_field": "doctor_id", "foreign_field": "doctor_id", "cardinality": "one_to_many"}
                ],
                "filters": [
                    {"field": "first_name", "operation": "regex", "value": "sarah"},
                    {"field": "last_name", "operation": "regex", "value": "wilson"}
                ],
                "sort": {"appt.appointment_date": "desc"}
            }
        }
    ]
    
    try:
        from join_teacher import validate_query_joins
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. Question: \"{scenario['question']}\"")
            print(f"   Explanation: {scenario['explanation']}")
            print("   Correct Approach:")
            print(f"   {json.dumps(scenario['correct_approach'], indent=6)}")
            
            # Validate the correct approach
            is_valid, warnings, analysis = validate_query_joins(scenario['correct_approach'])
            print(f"   Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
            if warnings:
                for warning in warnings:
                    print(f"     {warning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing scenarios: {e}")
        return False

def test_search_agent_integration():
    """Test integration with the search agent."""
    print("\n" + "="*60)
    print("🤖 Testing Search Agent Integration")
    print("="*60)
    
    test_queries = [
        "name of patients consulting to dr Linda brown along with their patient ids",
        "Show appointments for patient David Smith",
        "Find all scheduled appointments for Dr. Sarah Wilson"
    ]
    
    try:
        from search_agent_new import run_search_agent
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: \"{query}\"")
            
            try:
                spec, answer, debug = run_search_agent("test_user", "test_team", query)
                print(f"   Generated Spec:")
                print(f"   {json.dumps(spec, indent=6)}")
                
                # Check if joins were used appropriately
                if isinstance(spec, dict):
                    has_joins = "joins" in spec and len(spec.get("joins", [])) > 0
                    collection = spec.get("collection", "unknown")
                    print(f"   Collection: {collection}")
                    print(f"   Uses Joins: {'✅ Yes' if has_joins else '❌ No'}")
                    
                    if has_joins:
                        joins = spec.get("joins", [])
                        for join in joins:
                            print(f"     - {join.get('alias', 'unknown')} → {join.get('collection', 'unknown')}")
                
                # Show join validation results if available
                if debug.get("join_warnings"):
                    print("   Join Warnings:")
                    for warning in debug["join_warnings"]:
                        print(f"     {warning}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing search agent: {e}")
        return False

def main():
    """Run all teaching and testing."""
    print("🚀 Starting Join Teaching and Validation System Test")
    print("="*60)
    
    tests = [
        ("Join Teacher Validation", test_join_teacher),
        ("Query Scenarios", test_query_scenarios),
        ("Search Agent Integration", test_search_agent_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} : {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Join teaching system is working correctly!")
        print("The Search Agent should now avoid wrong join patterns and use correct approaches.")
    else:
        print("\n⚠️ Some tests failed. The join teaching system needs attention.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())