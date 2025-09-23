#!/usr/bin/env python3
"""
Test script demonstrating the difference between 'base_only' and 'all_involved' 
enforcement modes in multi-layer RLS authentication.

This shows how patient_id=P001 filtering works differently:
- base_only: Only applies to collections that directly have the patient_id field
- all_involved: Applies filtering through relationship traversal
"""

import json
from rls_engine import UserContext, RLSQueryInterceptor
from schema_registry import load_registry, get_rls_config, set_rls_config

def test_enforcement_modes():
    """Test both enforcement modes with the same scenario."""
    
    print("🏥 RLS Enforcement Mode Comparison")
    print("=" * 60)
    
    # Create user context
    user_context = UserContext(
        user_id="P001",
        role="patient",
        permissions=["read"]
    )
    
    # Test scenario: Query doctors collection with patient_id = P001
    test_query = {}
    rls_values = {"patient_id": "P001"}
    collection = "doctors"
    
    print(f"🔍 Test Scenario:")
    print(f"   Collection: {collection}")
    print(f"   RLS Values: {rls_values}")
    print(f"   Expected behavior: Patient P001 should only see their doctors")
    
    # Test 1: Base Only Enforcement
    print(f"\n1️⃣ BASE ONLY ENFORCEMENT")
    print("-" * 30)
    
    # Configure base_only enforcement
    rls_config = get_rls_config()
    rls_config["rls_layers"][0]["enforcement_mode"] = "base_only"
    
    rls_base_only = RLSQueryInterceptor(user_context, rls_config)
    enhanced_query_base = rls_base_only.enhance_find_query_with_multilayer(
        collection, test_query, rls_values
    )
    
    print(f"   Enhanced Query: {json.dumps(enhanced_query_base, indent=2)}")
    print(f"   Explanation: doctors collection has no patient_id field directly")
    print(f"   Result: Query unchanged - would return ALL doctors ❌")
    
    # Show audit log
    base_audit = rls_base_only.get_audit_log()
    for entry in base_audit[-2:]:
        print(f"   📝 {entry['action']}: {entry['message']}")
    
    # Test 2: All Involved Enforcement  
    print(f"\n2️⃣ ALL INVOLVED ENFORCEMENT")
    print("-" * 30)
    
    # Configure all_involved enforcement
    rls_config["rls_layers"][0]["enforcement_mode"] = "all_involved"
    
    rls_all_involved = RLSQueryInterceptor(user_context, rls_config)
    enhanced_query_all = rls_all_involved.enhance_find_query_with_multilayer(
        collection, test_query, rls_values
    )
    
    print(f"   Enhanced Query: {json.dumps(enhanced_query_all, indent=2)}")
    print(f"   Explanation: Traverses relationships to find doctors via appointments")
    print(f"   Expected Process:")
    print(f"     1. Find appointments where patient_id = P001")
    print(f"     2. Extract doctor_ids from those appointments")
    print(f"     3. Filter doctors by those doctor_ids")
    print(f"   Result: Only returns doctors associated with patient P001 ✅")
    
    # Show audit log
    all_audit = rls_all_involved.get_audit_log()
    for entry in all_audit[-3:]:
        print(f"   📝 {entry['action']}: {entry['message']}")

def demonstrate_relationship_logic():
    """Demonstrate the logical flow of relationship traversal."""
    
    print(f"\n🔗 Relationship Traversal Logic")
    print("=" * 40)
    
    print(f"Problem: Query doctors with patient_id = P001")
    print(f"")
    print(f"Schema Analysis:")
    print(f"  - doctors collection: Has doctor_id, but NO patient_id")
    print(f"  - appointments collection: Has BOTH patient_id AND doctor_id")
    print(f"  - Relationship: doctors ←→ appointments (via doctor_id)")
    print(f"")
    print(f"All Involved Process:")
    print(f"  Step 1: Check if 'doctors' has 'patient_id' field → NO")
    print(f"  Step 2: Look for relationships from 'doctors' collection")
    print(f"  Step 3: Find relationship: doctors ←→ appointments (doctor_id)")
    print(f"  Step 4: Check if 'appointments' has 'patient_id' field → YES")
    print(f"  Step 5: Query: db.appointments.find({{patient_id: 'P001'}}, {{doctor_id: 1}})")
    print(f"  Step 6: Extract doctor_ids: ['D001', 'D003', ...]")
    print(f"  Step 7: Apply filter: db.doctors.find({{doctor_id: {{$in: ['D001', 'D003', ...]}}}})")
    print(f"")
    print(f"Result: Patient P001 only sees doctors they have appointments with")

def show_configuration_example():
    """Show how to configure enforcement modes in schema_registry.json."""
    
    print(f"\n⚙️ Configuration Example")
    print("=" * 30)
    
    config_example = {
        "rls_config": {
            "enabled": True,
            "enforcement_mode": "all_involved",  # Global setting
            "rls_layers": [
                {
                    "field_name": "patient_id",
                    "display_name": "Patient ID",
                    "enabled": True,
                    "order": 1,
                    "enforcement_mode": "all_involved",  # Layer-specific override
                    "required": False,
                    "description": "Patient isolation with relationship traversal"
                }
            ],
            "collections": {
                "patients": {
                    "user_field": "patient_id",
                    "enforcement": "all_involved"
                },
                "appointments": {
                    "user_field": "patient_id",
                    "enforcement": "all_involved"
                },
                "doctors": {
                    "user_field": "doctor_id",  # Base field
                    "enforcement": "all_involved"  # But will use traversal for patient_id
                }
            }
        }
    }
    
    print("schema_registry.json configuration:")
    print(json.dumps(config_example, indent=2))

def test_with_multiple_layers():
    """Test with multiple RLS layers having different enforcement modes."""
    
    print(f"\n🔢 Multi-Layer Enforcement Test")
    print("=" * 40)
    
    # Create test configuration with mixed enforcement
    test_config = {
        "enabled": True,
        "enforcement_mode": "base_only",  # Global default
        "bypass_roles": ["admin"],
        "rls_layers": [
            {
                "field_name": "patient_id",
                "enabled": True,
                "order": 1,
                "enforcement_mode": "all_involved",  # Uses relationship traversal
                "required": False
            },
            {
                "field_name": "team_id", 
                "enabled": True,
                "order": 2,
                "enforcement_mode": "base_only",  # Direct field only
                "required": False
            }
        ],
        "collections": {}
    }
    
    user_context = UserContext(
        user_id="P001",
        role="patient", 
        permissions=["read"]
    )
    
    rls = RLSQueryInterceptor(user_context, test_config)
    
    # Test with both patient_id (all_involved) and team_id (base_only)
    rls_values = {
        "patient_id": "P001",  # Will trigger relationship traversal
        "team_id": "TEAM_A"    # Will only apply if doctors has team_id field
    }
    
    enhanced_query = rls.enhance_find_query_with_multilayer(
        "doctors", {}, rls_values
    )
    
    print(f"Multi-layer query result:")
    print(json.dumps(enhanced_query, indent=2))
    
    print(f"\nExpected behavior:")
    print(f"  - patient_id: Uses relationship traversal (all_involved)")
    print(f"  - team_id: Only applies if doctors collection has team_id field (base_only)")

if __name__ == "__main__":
    test_enforcement_modes()
    demonstrate_relationship_logic()
    show_configuration_example()
    test_with_multiple_layers()