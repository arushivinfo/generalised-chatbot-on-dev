#!/usr/bin/env python3
"""
AI-powered relationship suggester for MongoDB collections.
Uses LLM to analyze field names and suggest potential relationships.
"""

import os
import json
from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class AIRelationshipSuggester:
    def __init__(self):
        """Initialize the AI relationship suggester with OpenAI LLM."""
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
        
        self.suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert database architect analyzing MongoDB collections to suggest relationships.

Given collection schemas, identify potential foreign key relationships based on:
1. Field names that match (exact or similar)
2. Common naming patterns (e.g., user_id, customer_id, product_id)
3. Logical business relationships
4. Field types that match

For each suggested relationship, determine:
- Source collection and field
- Target collection and field  
- Cardinality (one_to_many, many_to_one, one_to_one)
- Join type (left, inner)
- Confidence level (high, medium, low)
- Reason for the suggestion

Return ONLY a JSON array with this structure:
[
  {{{{
    "source_collection": "collection1",
    "source_field": "field1", 
    "target_collection": "collection2",
    "target_field": "field2",
    "cardinality": "one_to_many|many_to_one|one_to_one",
    "join_type": "left",
    "confidence": "high|medium|low",
    "reason": "Explanation of why this relationship makes sense",
    "suggested_alias": "descriptive_alias"
  }}}}
]

Rules:
- Only suggest relationships between fields of compatible types
- Prefer exact field name matches over similar ones
- Consider business logic (users have orders, patients have appointments, etc.)
- Don't suggest self-referential relationships within same collection
- Focus on foreign key relationships, not just any field similarity"""),
            ("human", "Collections to analyze:\n\n{schema_data}")
        ])

    def analyze_collections(self, collections_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Analyze collections and suggest relationships.
        
        Args:
            collections_data: Dict with collection names as keys and field lists as values
            
        Returns:
            List of suggested relationships
        """
        try:
            # Debug: Print the input format
            print(f"AI Suggester received data: {type(collections_data)}")
            for k, v in collections_data.items():
                print(f"  {k}: {type(v)}")
                if isinstance(v, dict):
                    print(f"    keys: {list(v.keys())}")
                elif isinstance(v, list) and v:
                    print(f"    first item type: {type(v[0])}")
            
            # Format collection data for the LLM
            schema_text = self._format_collections_for_llm(collections_data)
            print(f"Formatted schema text:\n{schema_text[:500]}...")
            
            # Get suggestions from LLM
            response = self.llm.invoke(self.suggestion_prompt.format_messages(schema_data=schema_text))
            print(f"LLM response: {response.content[:500]}...")
            
            # Extract JSON from response (handle markdown code blocks)
            content = response.content.strip()
            
            # Try to extract JSON from code blocks first
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                # Fallback: use the entire content
                json_content = content
            
            print(f"Extracted JSON content: {json_content[:200]}...")
            
            # Parse JSON response with better error handling
            try:
                suggestions = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Full response content: {content}")
                print(f"Extracted JSON: {json_content}")
                
                # Try to find and extract just the JSON array part
                array_match = re.search(r'\[.*\]', json_content, re.DOTALL)
                if array_match:
                    try:
                        suggestions = json.loads(array_match.group(0))
                        print("Successfully parsed JSON from array match")
                    except json.JSONDecodeError:
                        print("Failed to parse even the extracted array")
                        return []
                else:
                    print("No JSON array found in content")
                    return []
            
            # Validate and clean suggestions
            validated_suggestions = self._validate_suggestions(suggestions, collections_data)
            
            return validated_suggestions
            
        except Exception as e:
            print(f"Error in AI relationship analysis: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _format_collections_for_llm(self, collections_data: Dict[str, List[Dict]]) -> str:
        """Format collection data for LLM analysis."""
        formatted = []
        
        for collection_name, collection_info in collections_data.items():
            formatted.append(f"Collection: {collection_name}")
            
            # Handle different data formats
            if isinstance(collection_info, dict):
                # New format: {"description": "...", "fields": [...]}
                description = collection_info.get('description', '')
                if description:
                    formatted.append(f"Description: {description}")
                
                fields = collection_info.get('fields', [])
            elif isinstance(collection_info, list):
                # Old format: direct list of fields
                fields = collection_info
            else:
                # Fallback
                fields = []
            
            formatted.append("Fields:")
            for field in fields:
                field_info = f"  - {field['name']} ({field['type']})"
                if field.get('description'):
                    field_info += f" - {field['description']}"
                formatted.append(field_info)
            
            formatted.append("")  # Empty line between collections
        
        return "\n".join(formatted)

    def _validate_suggestions(self, suggestions: List[Dict], collections_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Validate and clean AI suggestions."""
        validated = []
        
        for suggestion in suggestions:
            try:
                # Check required fields
                required_fields = ['source_collection', 'source_field', 'target_collection', 
                                 'target_field', 'cardinality', 'confidence', 'reason']
                
                if not all(field in suggestion for field in required_fields):
                    continue
                
                # Check collections exist
                if (suggestion['source_collection'] not in collections_data or 
                    suggestion['target_collection'] not in collections_data):
                    continue
                
                # Extract field names based on data format
                def get_field_names(collection_info):
                    if isinstance(collection_info, dict):
                        # New format: {"description": "...", "fields": [...]}
                        fields = collection_info.get('fields', [])
                    elif isinstance(collection_info, list):
                        # Old format: direct list of fields
                        fields = collection_info
                    else:
                        fields = []
                    return [f['name'] for f in fields]
                
                source_fields = get_field_names(collections_data[suggestion['source_collection']])
                target_fields = get_field_names(collections_data[suggestion['target_collection']])
                
                if (suggestion['source_field'] not in source_fields or 
                    suggestion['target_field'] not in target_fields):
                    continue
                
                # Set default values
                suggestion.setdefault('join_type', 'left')
                suggestion.setdefault('suggested_alias', suggestion['target_collection'])
                
                # Validate cardinality
                if suggestion['cardinality'] not in ['one_to_many', 'many_to_one', 'one_to_one']:
                    suggestion['cardinality'] = 'one_to_many'  # default
                
                # Validate confidence
                if suggestion['confidence'] not in ['high', 'medium', 'low']:
                    suggestion['confidence'] = 'medium'  # default
                
                validated.append(suggestion)
                
            except Exception as e:
                print(f"Error validating suggestion: {e}")
                continue
        
        return validated

    def suggest_for_collection_pair(self, source_collection: str, target_collection: str, 
                                   collections_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Suggest relationships between two specific collections.
        
        Args:
            source_collection: Name of source collection
            target_collection: Name of target collection
            collections_data: All collection data
            
        Returns:
            List of suggested relationships between the two collections
        """
        # Filter to only the two collections of interest
        filtered_data = {
            source_collection: collections_data[source_collection],
            target_collection: collections_data[target_collection]
        }
        
        # Get all suggestions and filter for this pair
        all_suggestions = self.analyze_collections(filtered_data)
        
        # Filter suggestions for this specific pair (both directions)
        pair_suggestions = [
            s for s in all_suggestions 
            if ((s['source_collection'] == source_collection and s['target_collection'] == target_collection) or
                (s['source_collection'] == target_collection and s['target_collection'] == source_collection))
        ]
        
        return pair_suggestions

def get_ai_relationship_suggestions(collections_data: Dict[str, Any]) -> List[Dict]:
    """
    Convenience function to get AI relationship suggestions.
    
    Args:
        collections_data: Dictionary with collection names as keys and field metadata as values
                         Can be either format:
                         - {"collection": [field_dicts...]} (old format)
                         - {"collection": {"description": "...", "fields": [field_dicts...]}} (new format)
        
    Returns:
        List of relationship suggestions
    """
    try:
        suggester = AIRelationshipSuggester()
        return suggester.analyze_collections(collections_data)
    except Exception as e:
        print(f"Failed to get AI suggestions: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Example usage
    sample_collections = {
        "patients": [
            {"name": "patient_id", "type": "string", "description": "Unique patient identifier"},
            {"name": "first_name", "type": "string", "description": "Patient first name"},
            {"name": "last_name", "type": "string", "description": "Patient last name"},
            {"name": "email", "type": "string", "description": "Patient email address"}
        ],
        "appointments": [
            {"name": "appointment_id", "type": "string", "description": "Unique appointment identifier"},
            {"name": "patient_id", "type": "string", "description": "Reference to patient"},
            {"name": "doctor_id", "type": "string", "description": "Reference to doctor"},
            {"name": "appointment_date", "type": "date", "description": "Date of appointment"}
        ],
        "doctors": [
            {"name": "doctor_id", "type": "string", "description": "Unique doctor identifier"},
            {"name": "first_name", "type": "string", "description": "Doctor first name"},
            {"name": "specialization", "type": "string", "description": "Medical specialty"}
        ]
    }
    
    suggestions = get_ai_relationship_suggestions(sample_collections)
    print("AI Relationship Suggestions:")
    print(json.dumps(suggestions, indent=2))