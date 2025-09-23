#!/usr/bin/env python3
"""
Example demonstrating enhanced "all_involved" RLS functionality.
Shows how patient_id=P001 filters doctors through appointments relationship.
"""

# Example Scenario: Patient P001 wants to see "their doctors"

# 1. BEFORE (base_only enforcement):
#    Query: db.doctors.find({})
#    Result: All doctors (no filtering because doctors collection has no patient_id field)

# 2. AFTER (all_involved enforcement with relationship traversal):
#    Query: db.doctors.find({})
#    RLS Process:
#    Step 1: Check if doctors collection has patient_id field -> NO
#    Step 2: Look for relationships from doctors to collections with patient_id
#    Step 3: Find: doctors -> appointments (via doctor_id)
#    Step 4: Find: appointments has patient_id field -> YES
#    Step 5: Query appointments for patient_id=P001 -> get doctor_ids
#    Step 6: Apply filter: db.doctors.find({"doctor_id": {"$in": ["D001", "D003"]}})

# Example with your schema:

def demonstrate_rls_flow():
    """Demonstrate the RLS flow for patient_id=P001 querying doctors."""
    
    # Simulated data
    appointments_data = [
        {"appointment_id": "A001", "patient_id": "P001", "doctor_id": "D001"},
        {"appointment_id": "A002", "patient_id": "P001", "doctor_id": "D003"},
        {"appointment_id": "A003", "patient_id": "P002", "doctor_id": "D002"},
        {"appointment_id": "A004", "patient_id": "P003", "doctor_id": "D001"}
    ]
    
    doctors_data = [
        {"doctor_id": "D001", "first_name": "Dr. John", "specialization": "Pediatrics"},
        {"doctor_id": "D002", "first_name": "Dr. Jane", "specialization": "Cardiology"},
        {"doctor_id": "D003", "first_name": "Dr. Sarah", "specialization": "Dermatology"}
    ]
    
    print("🏥 Hospital RLS Demonstration")
    print("=" * 50)
    
    print("\n📊 Sample Data:")
    print("Appointments:", appointments_data)
    print("Doctors:", doctors_data)
    
    print(f"\n🔒 RLS Filter: patient_id = P001")
    print(f"🔍 Query: Find doctors for patient P001")
    
    print(f"\n🛤️  RLS Traversal Process:")
    print(f"1. Check doctors collection for patient_id field -> ❌ Not found")
    print(f"2. Look for relationships from doctors collection")
    print(f"3. Find relationship: doctors <-> appointments (via doctor_id)")
    print(f"4. Check appointments for patient_id field -> ✅ Found")
    print(f"5. Query appointments where patient_id=P001")
    
    # Simulate the traversal
    patient_appointments = [apt for apt in appointments_data if apt["patient_id"] == "P001"]
    doctor_ids_for_patient = [apt["doctor_id"] for apt in patient_appointments]
    
    print(f"   → Found appointments: {patient_appointments}")
    print(f"   → Extracted doctor_ids: {doctor_ids_for_patient}")
    
    print(f"6. Apply filter to doctors: doctor_id IN {doctor_ids_for_patient}")
    
    # Final result
    filtered_doctors = [doc for doc in doctors_data if doc["doctor_id"] in doctor_ids_for_patient]
    
    print(f"\n✅ Final Result:")
    print(f"Patient P001 can see these doctors: {filtered_doctors}")
    
    print(f"\n🔄 Without RLS (all doctors): {len(doctors_data)} doctors")
    print(f"🔒 With RLS (filtered): {len(filtered_doctors)} doctors")
    
    return filtered_doctors

# Example configuration in your schema_registry.json:
example_config = {
    "rls_config": {
        "enabled": True,
        "enforcement_mode": "all_involved",  # This enables relationship traversal
        "rls_layers": [
            {
                "field_name": "patient_id",
                "display_name": "Patient ID",
                "enabled": True,
                "order": 1,
                "enforcement_mode": "all_involved",  # Key setting for traversal
                "required": True,
                "description": "Primary patient isolation with relationship traversal"
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
                "user_field": "doctor_id",  # Base field, but traversal will override
                "enforcement": "all_involved"
            }
        }
    }
}

if __name__ == "__main__":
    print("Configuration needed:")
    import json
    print(json.dumps(example_config, indent=2))
    print("\n" + "="*50)
    demonstrate_rls_flow()