#!/usr/bin/env python3
"""
Test script to verify enhanced RLS relationship traversal functionality.
Tests the scenario where patient_id=P001 should filter doctors through appointments.
"""

import json
from rls_engine import UserContext, RLSQueryInterceptor
from schema_registry import load_registry

def test_rls_relationship_traversal():
    """Test RLS relationship traversal for patient_id filtering."""
    
    print("🧪 Testing Enhanced RLS Relationship Traversal")
    print("=" * 60)
    
    # Create user context
    user_context = UserContext(
        user_id="P001",
        role="patient",
        permissions=["read"]
    )
    
    # Load registry to get RLS config
    reg = load_registry()
    
    # Create RLS interceptor
    rls = RLSQueryInterceptor(user_context, reg.get("rls_config"))
    
    # Test scenarios
    test_cases = [
        {
            "collection": "doctors",
            "query": {},
            "rls_values": {"patient_id": "P001"},
            "description": "Find doctors for patient P001 (should use relationship traversal)"
        },
        {
            "collection": "treatments", 
            "query": {},
            "rls_values": {"patient_id": "P001"},
            "description": "Find treatments for patient P001 (should traverse through appointments)"
        },
        {
            "collection": "patients",
            "query": {},
            "rls_values": {"patient_id": "P001"},
            "description": "Find patient P001 (should use direct filtering)"
        },
        {
            "collection": "appointments",
            "query": {},
            "rls_values": {"patient_id": "P001"},
            "description": "Find appointments for patient P001 (should use direct filtering)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"   Collection: {test_case['collection']}")
        print(f"   Original query: {test_case['query']}")
        print(f"   RLS values: {test_case['rls_values']}")
        
        try:
            # Apply RLS enhancement
            enhanced_query = rls.enhance_find_query_with_multilayer(
                test_case["collection"],
                test_case["query"],
                test_case["rls_values"]
            )
            
            print(f"   ✅ Enhanced query: {json.dumps(enhanced_query, indent=2)}")
            
            # Show audit log for this test
            audit_entries = rls.get_audit_log()
            recent_entries = audit_entries[-3:] if audit_entries else []
            for entry in recent_entries:
                print(f"   📝 Audit: {entry['action']} - {entry['message']}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Total audit entries: {len(rls.get_audit_log())}")
    print("\n" + "=" * 60)

def test_rls_schema_analysis():
    """Analyze the current schema for RLS relationships."""
    
    print("\n🔍 RLS Schema Analysis")
    print("=" * 40)
    
    reg = load_registry()
    collections = reg.get("collections", {})
    
    print(f"Total collections: {len(collections)}")
    
    for collection_name, collection_data in collections.items():
        print(f"\n📄 Collection: {collection_name}")
        
        # Show fields with patient_id
        fields = collection_data.get("fields", [])
        patient_fields = [f for f in fields if "patient_id" in f.get("name", "").lower()]
        if patient_fields:
            print(f"   ✅ Has patient_id field: {[f['name'] for f in patient_fields]}")
        else:
            print(f"   ❌ No patient_id field")
        
        # Show relationships
        relations = collection_data.get("relations", [])
        if relations:
            print(f"   🔗 Relationships ({len(relations)}):")
            for rel in relations:
                print(f"      - {rel.get('alias', 'N/A')} -> {rel.get('ref_collection', 'N/A')}")
                print(f"        {rel.get('local_field', 'N/A')} = {rel.get('foreign_field', 'N/A')}")
        else:
            print(f"   🔗 No relationships defined")

if __name__ == "__main__":
    test_rls_schema_analysis()
    test_rls_relationship_traversal()