#!/usr/bin/env python3
"""
Test script for AI relationship suggestions.
"""

import json
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_ai_relationship_suggestions():
    """Test the AI relationship suggestion functionality."""
    print("=== Testing AI Relationship Suggestions ===")
    
    # Sample collections data
    test_collections = {
        "patients": {
            "description": "Patient information and medical records",
            "fields": [
                {"name": "patient_id", "type": "string", "description": "Unique patient identifier"},
                {"name": "first_name", "type": "string", "description": "Patient first name"},
                {"name": "last_name", "type": "string", "description": "Patient last name"},
                {"name": "email", "type": "string", "description": "Patient email address"},
                {"name": "phone", "type": "string", "description": "Patient phone number"},
                {"name": "date_of_birth", "type": "date", "description": "Patient date of birth"}
            ]
        },
        "appointments": {
            "description": "Medical appointments and scheduling",
            "fields": [
                {"name": "appointment_id", "type": "string", "description": "Unique appointment identifier"},
                {"name": "patient_id", "type": "string", "description": "Patient who has the appointment"},
                {"name": "doctor_id", "type": "string", "description": "Doctor seeing the patient"},
                {"name": "appointment_date", "type": "date", "description": "Date and time of appointment"},
                {"name": "status", "type": "string", "description": "Appointment status"},
                {"name": "notes", "type": "string", "description": "Appointment notes"}
            ]
        },
        "doctors": {
            "description": "Doctor information and specialties",
            "fields": [
                {"name": "doctor_id", "type": "string", "description": "Unique doctor identifier"},
                {"name": "first_name", "type": "string", "description": "Doctor first name"},
                {"name": "last_name", "type": "string", "description": "Doctor last name"},
                {"name": "specialty", "type": "string", "description": "Medical specialty"},
                {"name": "email", "type": "string", "description": "Doctor email address"},
                {"name": "phone", "type": "string", "description": "Doctor phone number"}
            ]
        }
    }
    
    try:
        from ai_relationship_suggester import get_ai_relationship_suggestions
        
        print("Testing AI relationship suggestions...")
        suggestions = get_ai_relationship_suggestions(test_collections)
        
        print(f"\nAI Suggestions Generated:")
        print(json.dumps(suggestions, indent=2))
        
        # Validate suggestions structure
        for collection, relations in suggestions.items():
            print(f"\n{collection}:")
            for relation in relations:
                print(f"  - {relation['alias']} → {relation['ref_collection']}")
                print(f"    {relation['local_field']} = {relation['foreign_field']} ({relation['cardinality']})")
                if 'reasoning' in relation:
                    print(f"    Reasoning: {relation['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI suggestions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_relationship_management():
    """Test relationship management functions."""
    print("\n=== Testing Relationship Management ===")
    
    try:
        from schema_registry import (
            bulk_add_relationships, get_relationship_summary, 
            get_all_collection_relationships
        )
        
        # Test bulk add relationships
        test_relationships = {
            "patients": [
                {
                    "alias": "appointments",
                    "ref_collection": "appointments",
                    "local_field": "patient_id",
                    "foreign_field": "patient_id",
                    "cardinality": "one_to_many",
                    "join_type": "left"
                }
            ],
            "doctors": [
                {
                    "alias": "appointments",
                    "ref_collection": "appointments",
                    "local_field": "doctor_id",
                    "foreign_field": "doctor_id",
                    "cardinality": "one_to_many",
                    "join_type": "left"
                }
            ]
        }
        
        print("Testing bulk relationship addition...")
        result = bulk_add_relationships(test_relationships)
        print(f"Bulk add result: {result}")
        
        # Test relationship summary
        print("\nTesting relationship summary...")
        summary = get_relationship_summary()
        print(f"Summary: {json.dumps(summary, indent=2)}")
        
        # Test get all relationships
        print("\nTesting get all relationships...")
        all_rels = get_all_collection_relationships()
        print(f"All relationships: {json.dumps(all_rels, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing relationship management: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing AI Relationship Suggestions Implementation")
    
    # Test 1: AI Suggestions
    ai_test = test_ai_relationship_suggestions()
    
    # Test 2: Relationship Management
    mgmt_test = test_relationship_management()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"AI Suggestions Test: {'✅ PASS' if ai_test else '❌ FAIL'}")
    print(f"Management Functions Test: {'✅ PASS' if mgmt_test else '❌ FAIL'}")
    print(f"{'='*50}")
    
    if ai_test and mgmt_test:
        print("🎉 All tests passed! AI relationship suggestions are ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")