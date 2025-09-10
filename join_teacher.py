#!/usr/bin/env python3
"""
Join Teaching and Validation System
Teaches the Search Agent to avoid wrong joins and validates query logic.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from schema_registry import load_registry, get_all_fields, get_collection_relations

class JoinTeacher:
    """
    Educational system to teach correct join patterns and validate queries.
    """
    
    def __init__(self):
        self.registry = load_registry()
        self.all_fields = get_all_fields(self.registry)
        self.wrong_patterns = self._load_wrong_patterns()
        self.correct_patterns = self._load_correct_patterns()
    
    def _load_wrong_patterns(self) -> List[Dict]:
        """Load common wrong join patterns to avoid."""
        return [
            {
                "pattern": "direct_field_search",
                "description": "Searching for human names in ID fields",
                "example": {
                    "wrong": {
                        "collection": "appointments",
                        "filters": [{"field": "patient_id", "operation": "regex", "value": "david"}]
                    },
                    "why_wrong": "patient_id contains IDs like 'P001', not names like 'david'",
                    "correct_approach": "Join patients and appointments, filter by patient name"
                }
            },
            {
                "pattern": "reverse_join_direction",
                "description": "Wrong join direction for the query intent",
                "example": {
                    "wrong": {
                        "collection": "patients",
                        "joins": [{"alias": "doc", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id"}],
                        "filters": [{"owner": "doc", "field": "last_name", "operation": "regex", "value": "brown"}]
                    },
                    "why_wrong": "patients collection doesn't have direct doctor_id field",
                    "correct_approach": "Use appointments as intermediary: patients → appointments → doctors"
                }
            },
            {
                "pattern": "missing_intermediary",
                "description": "Trying to join collections without proper intermediary",
                "example": {
                    "wrong": {
                        "collection": "patients",
                        "joins": [{"alias": "doc", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id"}]
                    },
                    "why_wrong": "No direct relationship between patients and doctors",
                    "correct_approach": "Use appointments collection as bridge: patients → appointments → doctors"
                }
            }
        ]
    
    def _load_correct_patterns(self) -> List[Dict]:
        """Load correct join patterns with examples."""
        return [
            {
                "pattern": "name_to_appointments",
                "description": "Finding appointments by person name",
                "template": {
                    "patient_name_query": {
                        "collection": "patients",
                        "joins": [{"alias": "appt", "collection": "appointments", "local_field": "patient_id", "foreign_field": "patient_id", "cardinality": "one_to_many"}],
                        "filters": [{"field": "first_name", "operation": "regex", "value": "{patient_name}"}],
                        "sort": {"appt.appointment_date": "desc"}
                    },
                    "doctor_name_query": {
                        "collection": "doctors", 
                        "joins": [{"alias": "appt", "collection": "appointments", "local_field": "doctor_id", "foreign_field": "doctor_id", "cardinality": "one_to_many"}],
                        "filters": [{"field": "first_name", "operation": "regex", "value": "{doctor_name}"}],
                        "sort": {"appt.appointment_date": "desc"}
                    }
                }
            },
            {
                "pattern": "three_way_join",
                "description": "Queries needing data from all three collections",
                "template": {
                    "collection": "appointments",
                    "joins": [
                        {"alias": "patient", "collection": "patients", "local_field": "patient_id", "foreign_field": "patient_id", "cardinality": "many_to_one"},
                        {"alias": "doctor", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id", "cardinality": "many_to_one"}
                    ],
                    "filters": [
                        {"owner": "patient", "field": "first_name", "operation": "regex", "value": "{patient_name}"},
                        {"owner": "doctor", "field": "first_name", "operation": "regex", "value": "{doctor_name}"}
                    ]
                }
            }
        ]
    
    def analyze_query_intent(self, query_spec: Dict) -> Dict[str, Any]:
        """
        Analyze query intent and detect potential join issues.
        
        Args:
            query_spec: The query specification from LLM
            
        Returns:
            Analysis results with warnings and suggestions
        """
        analysis = {
            "intent": self._detect_intent(query_spec),
            "issues": [],
            "suggestions": [],
            "confidence": "high"
        }
        
        # Check for wrong patterns
        for wrong_pattern in self.wrong_patterns:
            if self._matches_wrong_pattern(query_spec, wrong_pattern):
                analysis["issues"].append({
                    "type": "wrong_pattern",
                    "pattern": wrong_pattern["pattern"],
                    "description": wrong_pattern["description"],
                    "why_wrong": wrong_pattern["example"]["why_wrong"],
                    "severity": "high"
                })
                analysis["confidence"] = "low"
        
        # Validate field relationships
        field_issues = self._validate_field_relationships(query_spec)
        analysis["issues"].extend(field_issues)
        
        # Generate suggestions
        analysis["suggestions"] = self._generate_suggestions(query_spec, analysis["issues"])
        
        return analysis
    
    def _detect_intent(self, query_spec: Dict) -> str:
        """Detect the intent of the query."""
        collection = query_spec.get("collection", "")
        filters = query_spec.get("filters", [])
        joins = query_spec.get("joins", [])
        
        if not joins:
            return "single_collection_query"
        
        # Analyze filter patterns
        has_name_filters = any(f.get("field") in ["first_name", "last_name", "name"] for f in filters)
        has_id_filters = any("_id" in f.get("field", "") for f in filters)
        
        if has_name_filters and collection in ["patients", "doctors"]:
            return "find_records_by_person_name"
        elif collection == "appointments" and len(joins) >= 2:
            return "comprehensive_appointment_query"
        else:
            return "complex_multi_collection_query"
    
    def _matches_wrong_pattern(self, query_spec: Dict, wrong_pattern: Dict) -> bool:
        """Check if query matches a known wrong pattern."""
        pattern_type = wrong_pattern["pattern"]
        
        if pattern_type == "direct_field_search":
            # Check if searching for names in ID fields
            filters = query_spec.get("filters", [])
            for f in filters:
                field_name = f.get("field", "")
                field_value = str(f.get("value", "")).lower()
                
                if "_id" in field_name and not field_value.startswith(("p", "d", "a")):
                    # Likely searching for name in ID field
                    if any(char.isalpha() for char in field_value) and not field_value.isdigit():
                        return True
        
        elif pattern_type == "reverse_join_direction":
            # Check for invalid join relationships
            joins = query_spec.get("joins", [])
            collection = query_spec.get("collection", "")
            
            for join in joins:
                local_field = join.get("local_field", "")
                foreign_field = join.get("foreign_field", "")
                target_collection = join.get("collection", "")
                
                # Check if the local collection actually has the local_field
                collection_fields = self.all_fields.get(collection, [])
                field_names = [f["name"] for f in collection_fields]
                
                if local_field not in field_names:
                    return True
        
        elif pattern_type == "missing_intermediary":
            # Check for direct joins that should use intermediary
            joins = query_spec.get("joins", [])
            collection = query_spec.get("collection", "")
            
            for join in joins:
                target_collection = join.get("collection", "")
                
                # Check if trying to directly join patients and doctors
                if ((collection == "patients" and target_collection == "doctors") or
                    (collection == "doctors" and target_collection == "patients")):
                    return True
        
        return False
    
    def _validate_field_relationships(self, query_spec: Dict) -> List[Dict]:
        """Validate that fields used in joins actually exist and are compatible."""
        issues = []
        collection = query_spec.get("collection", "")
        joins = query_spec.get("joins", [])
        
        collection_fields = self.all_fields.get(collection, [])
        base_field_names = [f["name"] for f in collection_fields]
        
        for join in joins:
            local_field = join.get("local_field", "")
            foreign_field = join.get("foreign_field", "")
            target_collection = join.get("collection", "")
            
            # Check if local field exists in base collection
            if local_field not in base_field_names:
                issues.append({
                    "type": "missing_field",
                    "description": f"Field '{local_field}' not found in collection '{collection}'",
                    "severity": "high",
                    "field": local_field,
                    "collection": collection
                })
            
            # Check if foreign field exists in target collection
            target_fields = self.all_fields.get(target_collection, [])
            target_field_names = [f["name"] for f in target_fields]
            
            if foreign_field not in target_field_names:
                issues.append({
                    "type": "missing_field",
                    "description": f"Field '{foreign_field}' not found in collection '{target_collection}'",
                    "severity": "high",
                    "field": foreign_field,
                    "collection": target_collection
                })
        
        return issues
    
    def _generate_suggestions(self, query_spec: Dict, issues: List[Dict]) -> List[Dict]:
        """Generate suggestions to fix query issues."""
        suggestions = []
        collection = query_spec.get("collection", "")
        
        for issue in issues:
            if issue["type"] == "wrong_pattern":
                pattern = issue["pattern"]
                
                if pattern == "direct_field_search":
                    suggestions.append({
                        "type": "fix_approach",
                        "description": "Use proper joins to connect collections",
                        "example": self._get_correct_pattern_example("name_to_appointments")
                    })
                
                elif pattern == "missing_intermediary":
                    suggestions.append({
                        "type": "add_intermediary",
                        "description": "Use appointments as bridge between patients and doctors",
                        "example": self._get_correct_pattern_example("three_way_join")
                    })
            
            elif issue["type"] == "missing_field":
                suggestions.append({
                    "type": "field_correction",
                    "description": f"Check field name '{issue['field']}' in collection '{issue['collection']}'",
                    "available_fields": [f["name"] for f in self.all_fields.get(issue['collection'], [])]
                })
        
        return suggestions
    
    def _get_correct_pattern_example(self, pattern_name: str) -> Dict:
        """Get example of correct pattern."""
        for pattern in self.correct_patterns:
            if pattern["pattern"] == pattern_name:
                return pattern["template"]
        return {}
    
    def teach_llm_examples(self) -> str:
        """Generate teaching examples for LLM system prompt."""
        examples = []
        
        examples.append("**CRITICAL JOIN DECISION RULES:**")
        examples.append("")
        
        # Add wrong vs right examples
        for wrong_pattern in self.wrong_patterns:
            examples.append(f"❌ **WRONG**: {wrong_pattern['description']}")
            examples.append(f"```json")
            examples.append(json.dumps(wrong_pattern["example"]["wrong"], indent=2))
            examples.append("```")
            examples.append(f"**Problem**: {wrong_pattern['example']['why_wrong']}")
            examples.append(f"**Solution**: {wrong_pattern['example']['correct_approach']}")
            examples.append("")
        
        # Add correct patterns
        examples.append("✅ **CORRECT PATTERNS:**")
        examples.append("")
        
        for correct_pattern in self.correct_patterns:
            examples.append(f"**{correct_pattern['description']}:**")
            if "patient_name_query" in correct_pattern["template"]:
                examples.append("```json")
                examples.append(json.dumps(correct_pattern["template"]["patient_name_query"], indent=2))
                examples.append("```")
            else:
                examples.append("```json")
                examples.append(json.dumps(correct_pattern["template"], indent=2))
                examples.append("```")
            examples.append("")
        
        return "\n".join(examples)

class QueryValidator:
    """
    Runtime validator for query specifications.
    """
    
    def __init__(self):
        self.teacher = JoinTeacher()
    
    def validate_query(self, query_spec: Dict) -> Tuple[bool, List[str], Dict]:
        """
        Validate query specification and provide feedback.
        
        Args:
            query_spec: Query specification from LLM
            
        Returns:
            (is_valid, warnings, analysis)
        """
        analysis = self.teacher.analyze_query_intent(query_spec)
        
        is_valid = True
        warnings = []
        
        # Check for high severity issues
        for issue in analysis["issues"]:
            if issue["severity"] == "high":
                is_valid = False
                warnings.append(f"❌ {issue['description']}")
            else:
                warnings.append(f"⚠️ {issue['description']}")
        
        # Add suggestions
        for suggestion in analysis["suggestions"]:
            warnings.append(f"💡 {suggestion['description']}")
        
        return is_valid, warnings, analysis
    
    def suggest_query_fix(self, query_spec: Dict) -> Optional[Dict]:
        """
        Suggest a corrected version of the query.
        
        Args:
            query_spec: Original query specification
            
        Returns:
            Suggested corrected query or None if no fix available
        """
        analysis = self.teacher.analyze_query_intent(query_spec)
        
        # For now, return basic fixes for common patterns
        if analysis["intent"] == "find_records_by_person_name":
            collection = query_spec.get("collection", "")
            filters = query_spec.get("filters", [])
            
            # Find name filters being applied to wrong collections
            for f in filters:
                if f.get("field") in ["first_name", "last_name"] and collection == "appointments":
                    # Suggest starting from patients/doctors instead
                    return {
                        "collection": "patients",  # or "doctors" based on context
                        "joins": [{"alias": "appt", "collection": "appointments", "local_field": "patient_id", "foreign_field": "patient_id", "cardinality": "one_to_many"}],
                        "filters": filters,
                        "sort": query_spec.get("sort", {}),
                        "limit": query_spec.get("limit", 20)
                    }
        
        return None

# Export key functions
def create_join_teacher() -> JoinTeacher:
    """Create a new JoinTeacher instance."""
    return JoinTeacher()

def validate_query_joins(query_spec: Dict) -> Tuple[bool, List[str], Dict]:
    """
    Validate query joins and provide feedback.
    
    Args:
        query_spec: Query specification to validate
        
    Returns:
        (is_valid, warnings, analysis)
    """
    validator = QueryValidator()
    return validator.validate_query(query_spec)

def get_join_teaching_examples() -> str:
    """Get teaching examples for LLM system prompt."""
    teacher = JoinTeacher()
    return teacher.teach_llm_examples()

if __name__ == "__main__":
    # Test the teaching system
    teacher = JoinTeacher()
    print("=== Join Teaching Examples ===")
    print(teacher.teach_llm_examples())
    
    # Test validation
    validator = QueryValidator()
    
    # Test a wrong query
    wrong_query = {
        "collection": "appointments",
        "filters": [{"field": "patient_id", "operation": "regex", "value": "david"}]
    }
    
    is_valid, warnings, analysis = validator.validate_query(wrong_query)
    print(f"\n=== Validation Test ===")
    print(f"Valid: {is_valid}")
    print("Warnings:")
    for warning in warnings:
        print(f"  {warning}")