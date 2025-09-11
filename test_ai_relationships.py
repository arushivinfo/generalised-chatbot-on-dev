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
    
    # Sample collections data with more complex relationships
    test_collections = {
        "patients": {
            "description": "Patient information and medical records",
            "fields": [
                {"name": "patient_id", "type": "string", "description": "Unique patient identifier"},
                {"name": "first_name", "type": "string", "description": "Patient first name"},
                {"name": "last_name", "type": "string", "description": "Patient last name"},
                {"name": "email", "type": "string", "description": "Patient email address"},
                {"name": "phone", "type": "string", "description": "Patient phone number"},
                {"name": "date_of_birth", "type": "date", "description": "Patient date of birth"},
                {"name": "primary_doctor_id", "type": "string", "description": "Primary care doctor ID"}
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
                {"name": "phone", "type": "string", "description": "Doctor phone number"},
                {"name": "department_id", "type": "string", "description": "Department ID"}
            ]
        },
        "treatments": {
            "description": "Medical treatments and procedures",
            "fields": [
                {"name": "treatment_id", "type": "string", "description": "Unique treatment identifier"},
                {"name": "patient_id", "type": "string", "description": "Patient receiving treatment"},
                {"name": "doctor_id", "type": "string", "description": "Doctor providing treatment"},
                {"name": "appointment_id", "type": "string", "description": "Related appointment"},
                {"name": "treatment_type", "type": "string", "description": "Type of treatment"},
                {"name": "cost", "type": "float", "description": "Treatment cost"}
            ]
        },
        "billing": {
            "description": "Patient billing and payments",
            "fields": [
                {"name": "billing_id", "type": "string", "description": "Unique billing identifier"},
                {"name": "patient_id", "type": "string", "description": "Patient being billed"},
                {"name": "treatment_id", "type": "string", "description": "Treatment being billed"},
                {"name": "amount", "type": "float", "description": "Billing amount"},
                {"name": "paid", "type": "boolean", "description": "Payment status"}
            ]
        }
    }
    
    try:
        from ai_relationship_suggester import get_ai_relationship_suggestions
        
        print("Testing comprehensive AI relationship suggestions...")
        suggestions = get_ai_relationship_suggestions(test_collections)
        
        print(f"\n🎯 AI Suggestions Generated: {len(suggestions)} total relationships")
        
        # Group by confidence level
        by_confidence = {'high': [], 'medium': [], 'low': []}
        for s in suggestions:
            conf = s.get('confidence', 'medium')
            if conf in by_confidence:
                by_confidence[conf].append(s)
        
        print(f"   🟢 High confidence: {len(by_confidence['high'])}")
        print(f"   🟡 Medium confidence: {len(by_confidence['medium'])}")
        print(f"   🔴 Low confidence: {len(by_confidence['low'])}")
        
        # Show sample relationships
        print(f"\n📋 Sample Relationships:")
        for i, suggestion in enumerate(suggestions[:10]):  # Show first 10
            source = suggestion.get('source_collection', 'Unknown')
            target = suggestion.get('target_collection', 'Unknown')
            source_field = suggestion.get('source_field', 'Unknown')
            target_field = suggestion.get('target_field', 'Unknown')
            confidence = suggestion.get('confidence', 'medium')
            cardinality = suggestion.get('cardinality', 'unknown')
            reason = suggestion.get('reason', 'No reason provided')
            
            conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(confidence, '⚪')
            print(f"   {i+1}. {conf_emoji} {source}.{source_field} → {target}.{target_field} ({cardinality})")
            print(f"      Reason: {reason}")
        
        if len(suggestions) > 10:
            print(f"   ... and {len(suggestions) - 10} more relationships")
        
        # Check for expected relationships
        expected_relationships = [
            ('patients', 'patient_id', 'appointments', 'patient_id'),
            ('doctors', 'doctor_id', 'appointments', 'doctor_id'),
            ('appointments', 'appointment_id', 'treatments', 'appointment_id'),
            ('patients', 'patient_id', 'treatments', 'patient_id'),
            ('patients', 'patient_id', 'billing', 'patient_id'),
            ('treatments', 'treatment_id', 'billing', 'treatment_id'),
        ]
        
        found_expected = []
        for expected in expected_relationships:
            for suggestion in suggestions:
                if (suggestion.get('source_collection') == expected[0] and
                    suggestion.get('source_field') == expected[1] and
                    suggestion.get('target_collection') == expected[2] and
                    suggestion.get('target_field') == expected[3]):
                    found_expected.append(expected)
                    break
        
        print(f"\n✅ Found {len(found_expected)}/{len(expected_relationships)} expected relationships")
        for expected in found_expected:
            print(f"   ✓ {expected[0]}.{expected[1]} → {expected[2]}.{expected[3]}")
        
        missing = [e for e in expected_relationships if e not in found_expected]
        if missing:
            print(f"\n⚠️ Missing expected relationships:")
            for missing_rel in missing:
                print(f"   ✗ {missing_rel[0]}.{missing_rel[1]} → {missing_rel[2]}.{missing_rel[3]}")
        
        return len(suggestions) > 0 and len(found_expected) >= len(expected_relationships) // 2
        
    except Exception as e:
        print(f"❌ Error testing AI suggestions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exhaustive_suggestions():
    """Test the exhaustive suggestion functionality specifically."""
    print("\n=== Testing Exhaustive AI Suggestions ===")
    
    # More comprehensive test data
    test_collections = {
        "users": {
            "description": "System users",
            "fields": [
                {"name": "user_id", "type": "string", "description": "Unique user identifier"},
                {"name": "username", "type": "string", "description": "User login name"},
                {"name": "email", "type": "string", "description": "User email"}
            ]
        },
        "orders": {
            "description": "Customer orders",
            "fields": [
                {"name": "order_id", "type": "string", "description": "Unique order identifier"},
                {"name": "user_id", "type": "string", "description": "User who placed the order"},
                {"name": "customer_id", "type": "string", "description": "Customer ID (alternative)"},
                {"name": "total", "type": "float", "description": "Order total"}
            ]
        },
        "order_items": {
            "description": "Items within orders", 
            "fields": [
                {"name": "item_id", "type": "string", "description": "Unique item identifier"},
                {"name": "order_id", "type": "string", "description": "Parent order"},
                {"name": "product_id", "type": "string", "description": "Product ordered"},
                {"name": "quantity", "type": "int", "description": "Quantity ordered"}
            ]
        },
        "products": {
            "description": "Product catalog",
            "fields": [
                {"name": "product_id", "type": "string", "description": "Unique product identifier"},
                {"name": "name", "type": "string", "description": "Product name"},
                {"name": "price", "type": "float", "description": "Product price"},
                {"name": "category_id", "type": "string", "description": "Product category"}
            ]
        }
    }
    
    try:
        from ai_relationship_suggester import AIRelationshipSuggester
        
        suggester = AIRelationshipSuggester()
        print("Testing exhaustive relationship detection...")
        
        # Test exhaustive suggestions
        suggestions = suggester.suggest_exhaustive_relationships(test_collections)
        
        print(f"\n🔍 Exhaustive Analysis Results:")
        print(f"   Total relationships found: {len(suggestions)}")
        
        # Group by analysis method (from reason field)
        analysis_methods = {}
        for s in suggestions:
            reason = s.get('reason', '')
            if 'pattern:' in reason.lower():
                method = 'Pattern-based'
            elif 'business logic:' in reason.lower():
                method = 'Business logic'
            elif 'exact' in reason.lower():
                method = 'Exact match'
            elif 'foreign key' in reason.lower():
                method = 'Foreign key'
            elif 'ai' in reason.lower():
                method = 'AI detected'
            else:
                method = 'Other'
            
            if method not in analysis_methods:
                analysis_methods[method] = []
            analysis_methods[method].append(s)
        
        print(f"\n📊 Relationship detection breakdown:")
        for method, relationships in analysis_methods.items():
            print(f"   {method}: {len(relationships)} relationships")
        
        # Show detailed relationships
        print(f"\n📋 All detected relationships:")
        for i, rel in enumerate(suggestions, 1):
            source = rel.get('source_collection')
            target = rel.get('target_collection')
            source_field = rel.get('source_field')
            target_field = rel.get('target_field')
            confidence = rel.get('confidence', 'medium')
            cardinality = rel.get('cardinality', 'unknown')
            
            conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(confidence, '⚪')
            print(f"   {i}. {conf_emoji} {source}.{source_field} → {target}.{target_field} ({cardinality})")
        
        # Verify we found expected comprehensive relationships
        expected_comprehensive = [
            ('users', 'user_id', 'orders', 'user_id'),
            ('orders', 'order_id', 'order_items', 'order_id'),
            ('products', 'product_id', 'order_items', 'product_id'),
            # Should also find reverse relationships
            ('orders', 'user_id', 'users', 'user_id'),
            ('order_items', 'order_id', 'orders', 'order_id'),
            ('order_items', 'product_id', 'products', 'product_id'),
        ]
        
        found = 0
        for expected in expected_comprehensive:
            for suggestion in suggestions:
                if (suggestion.get('source_collection') == expected[0] and
                    suggestion.get('source_field') == expected[1] and
                    suggestion.get('target_collection') == expected[2] and
                    suggestion.get('target_field') == expected[3]):
                    found += 1
                    break
        
        print(f"\n✅ Found {found}/{len(expected_comprehensive)} expected comprehensive relationships")
        
        return len(suggestions) >= 10  # Should find many relationships with exhaustive analysis
        
    except Exception as e:
        print(f"❌ Error testing exhaustive suggestions: {e}")
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
    
    # Test 3: Exhaustive Suggestions
    exhaustive_test = test_exhaustive_suggestions()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"AI Suggestions Test: {'✅ PASS' if ai_test else '❌ FAIL'}")
    print(f"Management Functions Test: {'✅ PASS' if mgmt_test else '❌ FAIL'}")
    print(f"Exhaustive Suggestions Test: {'✅ PASS' if exhaustive_test else '❌ FAIL'}")
    print(f"{'='*50}")
    
    if ai_test and mgmt_test and exhaustive_test:
        print("🎉 All tests passed! AI relationship suggestions are ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")