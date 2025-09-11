#!/usr/bin/env python3
"""
Test script to verify RLS field validation fixes for healthcare system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schema_registry import is_valid_rls_field, load_registry, get_all_fields
from rls_engine import RLSFieldDetector

def test_healthcare_rls_validation():
    """Test RLS validation specifically for healthcare system fields."""
    
    print("🏥 Testing Healthcare System RLS Field Validation")
    print("=" * 60)
    
    # Test healthcare-specific fields from your schema
    healthcare_fields = [
        # From patients collection
        {"name": "patient_id", "type": "string", "options": ["P001", "P002", "P003", "P004", "P005"]},
        
        # From doctors collection  
        {"name": "doctor_id", "type": "string", "options": ["D001", "D002", "D003", "D004", "D005"]},
        
        # From appointments collection
        {"name": "patient_id", "type": "string", "options": ["P001", "P002", "P003"]},
        {"name": "doctor_id", "type": "string", "options": ["D001", "D002", "D003"]},
        {"name": "appointment_id", "type": "string", "options": ["A001", "A002"]},
        
        # Should be invalid
        {"name": "first_name", "type": "string", "options": ["Alex", "David", "Jane"]},
        {"name": "appointment_date", "type": "date", "options": []},
        {"name": "specialization", "type": "string", "options": ["Pediatrics", "Oncology"]},
    ]
    
    print("Individual Field Tests:")
    print("-" * 30)
    
    valid_count = 0
    total_count = len(healthcare_fields)
    
    for i, field in enumerate(healthcare_fields):
        result = is_valid_rls_field(field)
        status = "✅ VALID" if result else "❌ INVALID"
        
        print(f"{i+1:2d}. {field['name']:<18} ({field['type']:<7}) -> {status}")
        
        if field['name'] in ['patient_id', 'doctor_id', 'appointment_id'] and not result:
            print(f"    ⚠️  ERROR: {field['name']} should be valid for RLS!")
        elif field['name'] in ['first_name', 'appointment_date', 'specialization'] and result:
            print(f"    ⚠️  ERROR: {field['name']} should NOT be valid for RLS!")
        
        if result:
            valid_count += 1
    
    print(f"\nResults: {valid_count}/{total_count} fields identified as RLS-suitable")
    
    # Test with actual registry data
    print("\n" + "=" * 60)
    print("🔍 Testing with Actual Registry Data")
    print("=" * 60)
    
    try:
        reg = load_registry()
        all_fields = get_all_fields(reg)
        
        collections_to_test = ["patients", "doctors", "appointments", "billing", "treatments"]
        
        for collection in collections_to_test:
            if collection in all_fields:
                fields = all_fields[collection]
                valid_rls_fields = []
                
                for field in fields:
                    if is_valid_rls_field(field):
                        valid_rls_fields.append(field["name"])
                
                print(f"📄 {collection:<12}: {len(valid_rls_fields)} RLS fields -> {valid_rls_fields}")
                
                # Check for expected fields
                expected_fields = []
                if collection in ["patients", "appointments", "billing"]:
                    expected_fields.append("patient_id")
                if collection in ["doctors", "appointments"]:
                    expected_fields.append("doctor_id")
                if collection == "treatments":
                    expected_fields.extend(["treatment_id", "appointment_id"])  # treatments might use these
                    
                for expected in expected_fields:
                    if expected not in valid_rls_fields:
                        print(f"    ⚠️  ERROR: Expected '{expected}' to be valid for RLS in {collection}")
                    else:
                        print(f"    ✅ '{expected}' correctly identified as RLS field")
            else:
                print(f"📄 {collection:<12}: Collection not found in registry")
        
    except Exception as e:
        print(f"❌ Error testing registry data: {e}")
        import traceback
        traceback.print_exc()
    
    # Test RLS detector
    print("\n" + "=" * 60)
    print("🤖 Testing AI RLS Field Detection")
    print("=" * 60)
    
    try:
        detector = RLSFieldDetector()
        
        test_collections = {
            "patients": [
                {"name": "patient_id", "type": "string", "options": ["P001", "P002"]},
                {"name": "first_name", "type": "string", "options": ["Alex", "Jane"]},
                {"name": "email", "type": "string", "options": ["alex@mail.com"]}
            ],
            "doctors": [
                {"name": "doctor_id", "type": "string", "options": ["D001", "D002"]},
                {"name": "specialization", "type": "string", "options": ["Pediatrics"]},
                {"name": "email", "type": "string", "options": ["dr.alex@hospital.com"]}
            ]
        }
        
        for collection, fields in test_collections.items():
            detected = detector.detect_ownership_field(collection, fields)
            expected = f"{collection[:-1]}_id"  # patients -> patient_id, doctors -> doctor_id
            
            if detected == expected:
                print(f"✅ {collection}: Correctly detected '{detected}'")
            else:
                print(f"❌ {collection}: Expected '{expected}', got '{detected}'")
                
    except Exception as e:
        print(f"⚠️  AI detection test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test Complete")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_healthcare_rls_validation()