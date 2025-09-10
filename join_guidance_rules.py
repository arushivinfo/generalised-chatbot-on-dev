# Enhanced Join Guidance Rules for Search Agent
JOIN_DECISION_RULES = """
CRITICAL JOIN DECISION RULES:

1. FIELD OWNERSHIP ANALYSIS:
   - Before creating filters, identify which collection contains each field
   - If a field exists in a joined collection, set "owner": "alias_name"  
   - If a field exists in the base collection, set "owner": null
   - NEVER filter on fields that don't exist in the chosen collection

2. COLLECTION SELECTION LOGIC:
   For queries about relationships, choose the collection that:
   - Contains the primary entities being requested
   - Serves as the natural "hub" for the required joins
   - Minimizes the number of join hops needed

3. COMMON PATTERNS:
   a) "Patients with X condition/treatment" → Start from appointments or treatments
   b) "Doctors treating X" → Start from appointments  
   c) "Appointments for X criteria" → Start from appointments
   d) "Bills/costs for X" → Start from billing or treatments

4. JOIN VALIDATION:
   - Verify all joins are defined in the schema relationships
   - Ensure proper cardinality (one_to_many vs many_to_one)
   - Use appropriate join types (inner vs left)

5. FIELD REFERENCE EXAMPLES:
   WRONG: 
   {
     "field": "treatment_type",
     "owner": null,  // ❌ treatment_type is NOT in appointments
     "collection": "appointments"
   }
   
   RIGHT:
   {
     "field": "treatment_type", 
     "owner": "treatments",  // ✅ treatment_type IS in treatments
     "collection": "appointments",
     "joins": [{"alias": "treatments", "collection": "treatments", ...}]
   }

6. LOGICAL FLOW VERIFICATION:
   Before finalizing query, check:
   - Does the base collection make logical sense?
   - Are all filter fields available in their specified collections?
   - Do the joins create a valid path to all required data?
   - Would a different base collection be more efficient?
"""

# Common Query Pattern Solutions
QUERY_PATTERN_SOLUTIONS = {
    "patients_with_treatment": {
        "description": "Find patients who received specific treatments",
        "recommended_approach": "Start from appointments → join treatments + patients",
        "example": {
            "collection": "appointments",
            "joins": [
                {"alias": "treatment", "collection": "treatments", "local_field": "appointment_id", "foreign_field": "appointment_id"},
                {"alias": "patient", "collection": "patients", "local_field": "patient_id", "foreign_field": "patient_id"}
            ],
            "filters": [
                {"field": "treatment_type", "operation": "regex", "value": "Chemotherapy", "owner": "treatment"}
            ]
        }
    },
    "doctors_treating_condition": {
        "description": "Find doctors who treat specific conditions",
        "recommended_approach": "Start from appointments → join doctors + treatments", 
        "example": {
            "collection": "appointments",
            "joins": [
                {"alias": "doctor", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id"},
                {"alias": "treatment", "collection": "treatments", "local_field": "appointment_id", "foreign_field": "appointment_id"}
            ],
            "filters": [
                {"field": "treatment_type", "operation": "regex", "value": "Chemotherapy", "owner": "treatment"}
            ]
        }
    },
    "patient_doctor_relationships": {
        "description": "Find patients consulting specific doctors",
        "recommended_approach": "Start from appointments → join patients + doctors",
        "example": {
            "collection": "appointments", 
            "joins": [
                {"alias": "patient", "collection": "patients", "local_field": "patient_id", "foreign_field": "patient_id"},
                {"alias": "doctor", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id"}
            ],
            "filters": [
                {"field": "first_name", "operation": "regex", "value": "Linda", "owner": "doctor"},
                {"field": "last_name", "operation": "regex", "value": "Brown", "owner": "doctor"}
            ]
        }
    }
}