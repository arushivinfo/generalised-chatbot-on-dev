#!/usr/bin/env python3
"""
AI-powered relationship suggester for MongoDB collections.
Uses LLM to analyze field names and suggest potential relationships.
"""

import os
import json
from typing import Dict, List, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from llm_services import call_admin_ai_model
load_dotenv()

class AIRelationshipSuggester:
    def __init__(self):
        """Initialize the AI relationship suggester."""
        self.suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert database architect analyzing MongoDB collections to suggest comprehensive relationships.

CRITICAL: Your task is to identify ALL POSSIBLE relationships between collections, not just the most obvious ones.

Analyze based on:
1. **Exact field name matches** (e.g., user_id ↔ user_id, patient_id ↔ patient_id)
2. **Similar field patterns** (e.g., customer_id ↔ cust_id, product_id ↔ prod_id)
3. **Common naming conventions** (e.g., *_id fields, foreign key patterns)
4. **Business logic relationships** (patients → appointments → treatments, orders → order_items)
5. **Cross-referential fields** (created_by, assigned_to, managed_by)
6. **Hierarchical relationships** (parent_id, category_id)

IMPORTANT RULES:
- Generate EVERY POSSIBLE relationship, even if it seems redundant
- Include BOTH directions when applicable (A→B AND B→A as separate relationships)
- Don't worry about duplicates - provide all relationships you can identify
- Include low-confidence relationships - let the user decide
- Consider indirect relationships through intermediate fields
- Look for junction table patterns (many-to-many relationships)

For EACH relationship found, determine:
- Source collection and field
- Target collection and field  
- Cardinality (one_to_many, many_to_one, one_to_one, many_to_many)
- Join type (left, inner)
- Confidence level (high, medium, low)
- Detailed reason explaining the relationship logic
- Suggested descriptive alias

RESPONSE FORMAT - Return ONLY a JSON array:
[
  {{{{
    "source_collection": "collection1",
    "source_field": "field1", 
    "target_collection": "collection2",
    "target_field": "field2",
    "cardinality": "one_to_many|many_to_one|one_to_one|many_to_many",
    "join_type": "left|inner",
    "confidence": "high|medium|low",
    "reason": "Detailed explanation of the relationship logic and business case",
    "suggested_alias": "descriptive_alias_name"
  }}}}
]

COMPREHENSIVE ANALYSIS CHECKLIST:
✓ Exact field name matches across all collections
✓ Pattern matches (variations of same field name)
✓ Foreign key relationships (*_id fields)
✓ Business entity relationships (user → profile, order → items)
✓ Audit trail fields (created_by, updated_by)
✓ Hierarchical relationships (parent_child, category_subcategory)  
✓ Reference tables and lookup relationships
✓ Many-to-many junction patterns
✓ Temporal relationships (date-based joins)
✓ Cross-collection field similarities

GENERATE ALL POSSIBILITIES - Don't filter out relationships that might seem obvious or redundant."""),
            ("human", "Collections to analyze comprehensively:\n\n{schema_data}")
        ])

    def analyze_collections(self, collections_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Analyze collections and suggest comprehensive relationships.
        
        Args:
            collections_data: Dict with collection names as keys and field lists as values
            
        Returns:
            List of suggested relationships
        """
        try:
            # Debug: Print the input format
            print(f"🔍 AI Suggester analyzing {len(collections_data)} collections")
            for k, v in collections_data.items():
                if isinstance(v, dict):
                    field_count = len(v.get('fields', []))
                    print(f"  📄 {k}: {field_count} fields")
                else:
                    print(f"  📄 {k}: {len(v) if isinstance(v, list) else 'unknown'} items")
            
            # Pre-analyze for potential relationships before sending to LLM
            potential_relationships = self._identify_potential_relationships(collections_data)
            print(f"🎯 Pre-identified {len(potential_relationships)} potential relationship patterns")
            
            # Format collection data for the LLM with enhanced context
            schema_text = self._format_collections_for_llm_comprehensive(collections_data, potential_relationships)
            print(f"📝 Generated comprehensive schema text: {len(schema_text)} characters")
            
            # Get suggestions from LLM with retry logic
            max_retries = 2
            suggestions = []
            
            for attempt in range(max_retries + 1):
                try:
                    print(f"🤖 LLM Analysis attempt {attempt + 1}/{max_retries + 1}")
                    
                    # Format messages for the new model call
                    formatted_messages = self.suggestion_prompt.format_messages(schema_data=schema_text)
                    messages_for_llm = [{"role": msg.type, "content": msg.content} for msg in formatted_messages]
                    
                    response_content = call_admin_ai_model(messages_for_llm)
                    
                    # Extract and parse JSON
                    suggestions = self._extract_and_parse_json(response_content)
                    
                    if suggestions:
                        print(f"✅ LLM returned {len(suggestions)} raw suggestions")
                        break
                    else:
                        print(f"⚠️ No suggestions from attempt {attempt + 1}")
                        
                except Exception as e:
                    print(f"❌ LLM attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries:
                        # Fallback to rule-based suggestions
                        print("🔧 Falling back to rule-based relationship detection")
                        suggestions = self._generate_fallback_relationships(collections_data)
            
            # Validate and enhance suggestions
            validated_suggestions = self._validate_and_enhance_suggestions(suggestions, collections_data)
            
            # Add bidirectional relationships where appropriate
            bidirectional_suggestions = self._add_bidirectional_relationships(validated_suggestions)
            
            print(f"🎉 Final result: {len(bidirectional_suggestions)} comprehensive relationship suggestions")
            return bidirectional_suggestions
            
        except Exception as e:
            print(f"💥 Critical error in AI relationship analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback to rule-based approach
            try:
                return self._generate_fallback_relationships(collections_data)
            except:
                return []

    def _identify_potential_relationships(self, collections_data: Dict[str, Any]) -> List[Dict]:
        """Pre-analyze collections to identify potential relationships using rule-based approach."""
        relationships = []
        
        # Extract field information from collections
        collection_fields = {}
        for collection_name, collection_info in collections_data.items():
            if isinstance(collection_info, dict):
                fields = collection_info.get('fields', [])
            else:
                fields = collection_info if isinstance(collection_info, list) else []
            
            collection_fields[collection_name] = {
                field['name']: field for field in fields
            }
        
        # Find all potential relationships
        for source_collection, source_fields in collection_fields.items():
            for target_collection, target_fields in collection_fields.items():
                if source_collection == target_collection:
                    continue  # Skip self-references
                
                # Check for exact field name matches
                for source_field_name, source_field in source_fields.items():
                    for target_field_name, target_field in target_fields.items():
                        # Exact match
                        if source_field_name == target_field_name and source_field_name.endswith('_id'):
                            relationships.append({
                                'source_collection': source_collection,
                                'source_field': source_field_name,
                                'target_collection': target_collection,
                                'target_field': target_field_name,
                                'match_type': 'exact',
                                'confidence': 'high'
                            })
                        
                        # Pattern matches (similar field names)
                        elif self._are_similar_id_fields(source_field_name, target_field_name):
                            relationships.append({
                                'source_collection': source_collection,
                                'source_field': source_field_name,
                                'target_collection': target_collection,
                                'target_field': target_field_name,
                                'match_type': 'pattern',
                                'confidence': 'medium'
                            })
                        
                        # Cross-reference fields (one collection's ID in another)
                        elif source_field_name == f"{target_collection[:-1]}_id" or source_field_name == f"{target_collection}_id":
                            relationships.append({
                                'source_collection': source_collection,
                                'source_field': source_field_name,
                                'target_collection': target_collection,
                                'target_field': 'id',  # Assume primary key
                                'match_type': 'foreign_key',
                                'confidence': 'high'
                            })
        
        return relationships

    def _are_similar_id_fields(self, field1: str, field2: str) -> bool:
        """Check if two field names represent similar ID fields."""
        # Common ID field patterns
        id_patterns = [
            ('user_id', 'usr_id'), ('customer_id', 'cust_id'), ('product_id', 'prod_id'),
            ('account_id', 'acct_id'), ('employee_id', 'emp_id'), ('patient_id', 'pat_id'),
            ('doctor_id', 'doc_id'), ('appointment_id', 'appt_id'), ('treatment_id', 'treat_id')
        ]
        
        field1_lower = field1.lower()
        field2_lower = field2.lower()
        
        for pattern1, pattern2 in id_patterns:
            if (field1_lower == pattern1 and field2_lower == pattern2) or \
               (field1_lower == pattern2 and field2_lower == pattern1):
                return True
        
        return False

    def _format_collections_for_llm_comprehensive(self, collections_data: Dict[str, Any], 
                                                potential_relationships: List[Dict]) -> str:
        """Format collection data with potential relationships for comprehensive LLM analysis."""
        formatted = []
        
        # Add collection schemas
        for collection_name, collection_info in collections_data.items():
            formatted.append(f"## Collection: {collection_name}")
            
            # Handle different data formats
            if isinstance(collection_info, dict):
                description = collection_info.get('description', '')
                if description:
                    formatted.append(f"**Description**: {description}")
                fields = collection_info.get('fields', [])
            else:
                fields = collection_info if isinstance(collection_info, list) else []
            
            formatted.append("**Fields**:")
            for field in fields:
                field_info = f"  - **{field['name']}** ({field['type']})"
                if field.get('description'):
                    field_info += f" - {field['description']}"
                if field.get('options'):
                    sample_options = field['options'][:3]  # Show first 3 options
                    field_info += f" (sample values: {', '.join(map(str, sample_options))})"
                formatted.append(field_info)
            formatted.append("")
        
        # Add pre-identified relationships as hints
        if potential_relationships:
            formatted.append("## Pre-identified Relationship Patterns:")
            for rel in potential_relationships[:10]:  # Show first 10
                formatted.append(f"- {rel['source_collection']}.{rel['source_field']} ↔ {rel['target_collection']}.{rel['target_field']} ({rel['match_type']}, {rel['confidence']} confidence)")
            formatted.append("")
        
        # Add comprehensive analysis instructions
        formatted.append("## Analysis Instructions:")
        formatted.append("Please identify ALL possible relationships including:")
        formatted.append("1. Direct foreign key relationships (exact field matches)")
        formatted.append("2. Similar field patterns with variations")
        formatted.append("3. Business logic relationships (users→orders, patients→appointments)")
        formatted.append("4. Cross-reference relationships (created_by, assigned_to)")
        formatted.append("5. Hierarchical relationships (parent_id, category_id)")
        formatted.append("6. Junction table patterns (many-to-many)")
        formatted.append("7. BOTH directions where applicable (A→B AND B→A)")
        formatted.append("")
        
        return "\n".join(formatted)

    def _extract_and_parse_json(self, content: str) -> List[Dict]:
        """Extract and parse JSON from LLM response with multiple fallback strategies."""
        import re
        
        # Strategy 1: Try to extract from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for JSON array pattern
        array_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try parsing the entire content
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Try to clean and parse
        cleaned = content.strip().strip('`').strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        print(f"⚠️ Failed to parse JSON from LLM response: {content[:200]}...")
        return []

    def _validate_and_enhance_suggestions(self, suggestions: List[Dict], 
                                        collections_data: Dict[str, Any]) -> List[Dict]:
        """Validate and enhance AI suggestions with additional metadata."""
        validated = []
        
        # Get field information for validation
        collection_fields = {}
        for collection_name, collection_info in collections_data.items():
            if isinstance(collection_info, dict):
                fields = collection_info.get('fields', [])
            else:
                fields = collection_info if isinstance(collection_info, list) else []
            
            collection_fields[collection_name] = [f['name'] for f in fields]
        
        for suggestion in suggestions:
            try:
                # Validate required fields
                required_fields = ['source_collection', 'source_field', 'target_collection', 
                                 'target_field', 'cardinality']
                
                if not all(field in suggestion for field in required_fields):
                    print(f"⚠️ Skipping suggestion missing required fields: {suggestion}")
                    continue
                
                # Validate collections exist
                if (suggestion['source_collection'] not in collection_fields or 
                    suggestion['target_collection'] not in collection_fields):
                    print(f"⚠️ Skipping suggestion with invalid collections: {suggestion}")
                    continue
                
                # Validate fields exist
                source_fields = collection_fields[suggestion['source_collection']]
                target_fields = collection_fields[suggestion['target_collection']]
                
                if (suggestion['source_field'] not in source_fields or 
                    suggestion['target_field'] not in target_fields):
                    print(f"⚠️ Skipping suggestion with invalid fields: {suggestion}")
                    continue
                
                # Set defaults and normalize
                suggestion.setdefault('join_type', 'left')
                suggestion.setdefault('confidence', 'medium')
                suggestion.setdefault('reason', 'AI identified relationship')
                suggestion.setdefault('suggested_alias', suggestion['target_collection'])
                
                # Normalize cardinality
                if suggestion['cardinality'] not in ['one_to_many', 'many_to_one', 'one_to_one', 'many_to_many']:
                    suggestion['cardinality'] = 'one_to_many'  # default
                
                validated.append(suggestion)
                
            except Exception as e:
                print(f"⚠️ Error validating suggestion: {e}, suggestion: {suggestion}")
                continue
        
        return validated

    def _add_bidirectional_relationships(self, suggestions: List[Dict]) -> List[Dict]:
        """Add reverse relationships where appropriate to ensure comprehensive coverage."""
        bidirectional = list(suggestions)  # Start with original suggestions
        
        for suggestion in suggestions:
            # Create reverse relationship with appropriate cardinality flip
            reverse_cardinality_map = {
                'one_to_many': 'many_to_one',
                'many_to_one': 'one_to_many',
                'one_to_one': 'one_to_one',
                'many_to_many': 'many_to_many'
            }
            
            reverse_suggestion = {
                'source_collection': suggestion['target_collection'],
                'source_field': suggestion['target_field'],
                'target_collection': suggestion['source_collection'],
                'target_field': suggestion['source_field'],
                'cardinality': reverse_cardinality_map.get(suggestion['cardinality'], 'one_to_many'),
                'join_type': suggestion.get('join_type', 'left'),
                'confidence': 'medium',  # Slightly lower confidence for reverse
                'reason': f"Reverse of: {suggestion.get('reason', 'AI relationship')}",
                'suggested_alias': suggestion['source_collection']
            }
            
            # Check if reverse relationship doesn't already exist
            exists = any(
                existing['source_collection'] == reverse_suggestion['source_collection'] and
                existing['target_collection'] == reverse_suggestion['target_collection'] and
                existing['source_field'] == reverse_suggestion['source_field'] and
                existing['target_field'] == reverse_suggestion['target_field']
                for existing in bidirectional
            )
            
            if not exists:
                bidirectional.append(reverse_suggestion)
        
        return bidirectional

    def _generate_fallback_relationships(self, collections_data: Dict[str, Any]) -> List[Dict]:
        """Generate relationships using rule-based approach as fallback."""
        relationships = []
        
        # Get field information
        collection_fields = {}
        for collection_name, collection_info in collections_data.items():
            if isinstance(collection_info, dict):
                fields = collection_info.get('fields', [])
            else:
                fields = collection_info if isinstance(collection_info, list) else []
            
            collection_fields[collection_name] = fields
        
        # Generate relationships based on common patterns
        for source_collection, source_fields in collection_fields.items():
            for target_collection, target_fields in collection_fields.items():
                if source_collection == target_collection:
                    continue
                
                # Look for foreign key patterns
                for source_field in source_fields:
                    source_name = source_field['name']
                    
                    # Check if this field looks like a foreign key to target collection
                    if (source_name == f"{target_collection[:-1]}_id" or  # singular
                        source_name == f"{target_collection}_id" or       # plural
                        source_name == target_collection.lower() + "_id"):
                        
                        relationships.append({
                            'source_collection': source_collection,
                            'source_field': source_name,
                            'target_collection': target_collection,
                            'target_field': f"{target_collection[:-1]}_id" if target_collection.endswith('s') else f"{target_collection}_id",
                            'cardinality': 'many_to_one',
                            'join_type': 'left',
                            'confidence': 'high',
                            'reason': f"Foreign key pattern: {source_name} references {target_collection}",
                            'suggested_alias': target_collection
                        })
        
        return relationships

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

    def suggest_exhaustive_relationships(self, collections_data: Dict[str, Any]) -> List[Dict]:
        """
        Generate exhaustive relationship suggestions using multiple strategies.
        This method combines AI analysis with rule-based detection for maximum coverage.
        """
        try:
            print("🔍 Starting exhaustive relationship analysis...")
            
            all_suggestions = []
            
            # Strategy 1: Rule-based relationship detection
            print("📋 Running rule-based analysis...")
            rule_based = self._generate_comprehensive_rule_based_relationships(collections_data)
            print(f"   Found {len(rule_based)} rule-based relationships")
            all_suggestions.extend(rule_based)
            
            # Strategy 2: AI-powered analysis
            print("🤖 Running AI-powered analysis...")
            ai_based = self.analyze_collections(collections_data)
            print(f"   Found {len(ai_based)} AI-suggested relationships")
            all_suggestions.extend(ai_based)
            
            # Strategy 3: Pattern-based detection
            print("🔍 Running pattern-based analysis...")
            pattern_based = self._generate_pattern_based_relationships(collections_data)
            print(f"   Found {len(pattern_based)} pattern-based relationships")
            all_suggestions.extend(pattern_based)
            
            # Strategy 4: Business logic relationships
            print("💼 Running business logic analysis...")
            business_logic = self._generate_business_logic_relationships(collections_data)
            print(f"   Found {len(business_logic)} business logic relationships")
            all_suggestions.extend(business_logic)
            
            # Remove duplicates while preserving all unique relationships
            unique_suggestions = self._remove_duplicate_relationships(all_suggestions)
            print(f"🎯 Total unique relationships: {len(unique_suggestions)}")
            
            return unique_suggestions
            
        except Exception as e:
            print(f"💥 Error in exhaustive analysis: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _generate_comprehensive_rule_based_relationships(self, collections_data: Dict[str, Any]) -> List[Dict]:
        """Generate comprehensive rule-based relationships."""
        relationships = []
        
        # Extract field information
        collection_fields = {}
        for collection_name, collection_info in collections_data.items():
            if isinstance(collection_info, dict):
                fields = collection_info.get('fields', [])
            else:
                fields = collection_info if isinstance(collection_info, list) else []
            
            collection_fields[collection_name] = {f['name']: f for f in fields}
        
        # Generate all possible relationships
        for source_coll, source_fields in collection_fields.items():
            for target_coll, target_fields in collection_fields.items():
                if source_coll == target_coll:
                    continue
                
                for source_field_name, source_field in source_fields.items():
                    for target_field_name, target_field in target_fields.items():
                        
                        # Rule 1: Exact field name matches
                        if source_field_name == target_field_name and source_field_name.endswith('_id'):
                            relationships.append({
                                'source_collection': source_coll,
                                'source_field': source_field_name,
                                'target_collection': target_coll,
                                'target_field': target_field_name,
                                'cardinality': 'many_to_one',
                                'join_type': 'left',
                                'confidence': 'high',
                                'reason': f"Exact ID field match: {source_field_name}",
                                'suggested_alias': target_coll
                            })
                        
                        # Rule 2: Foreign key patterns
                        elif source_field_name == f"{target_coll[:-1]}_id" or source_field_name == f"{target_coll}_id":
                            relationships.append({
                                'source_collection': source_coll,
                                'source_field': source_field_name,
                                'target_collection': target_coll,
                                'target_field': f"{target_coll[:-1]}_id",
                                'cardinality': 'many_to_one',
                                'join_type': 'left',
                                'confidence': 'high',
                                'reason': f"Foreign key pattern: {source_field_name} → {target_coll}",
                                'suggested_alias': target_coll
                            })
                        
                        # Rule 3: Similar ID field names
                        elif self._are_similar_id_fields(source_field_name, target_field_name):
                            relationships.append({
                                'source_collection': source_coll,
                                'source_field': source_field_name,
                                'target_collection': target_coll,
                                'target_field': target_field_name,
                                'cardinality': 'many_to_one',
                                'join_type': 'left',
                                'confidence': 'medium',
                                'reason': f"Similar ID fields: {source_field_name} ≈ {target_field_name}",
                                'suggested_alias': target_coll
                            })
        
        return relationships

    def _generate_pattern_based_relationships(self, collections_data: Dict[str, Any]) -> List[Dict]:
        """Generate relationships based on common patterns."""
        relationships = []
        
        # Common relationship patterns
        patterns = [
            # Healthcare patterns
            ('patients', 'patient_id', 'appointments', 'patient_id', 'one_to_many'),
            ('patients', 'patient_id', 'treatments', 'patient_id', 'one_to_many'),
            ('patients', 'patient_id', 'billing', 'patient_id', 'one_to_many'),
            ('doctors', 'doctor_id', 'appointments', 'doctor_id', 'one_to_many'),
            ('doctors', 'doctor_id', 'treatments', 'doctor_id', 'one_to_many'),
            ('appointments', 'appointment_id', 'treatments', 'appointment_id', 'one_to_one'),
            
            # E-commerce patterns
            ('customers', 'customer_id', 'orders', 'customer_id', 'one_to_many'),
            ('customers', 'customer_id', 'reviews', 'customer_id', 'one_to_many'),
            ('products', 'product_id', 'order_items', 'product_id', 'one_to_many'),
            ('products', 'product_id', 'reviews', 'product_id', 'one_to_many'),
            ('orders', 'order_id', 'order_items', 'order_id', 'one_to_many'),
            ('categories', 'category_id', 'products', 'category_id', 'one_to_many'),
            
            # General business patterns
            ('users', 'user_id', 'orders', 'user_id', 'one_to_many'),
            ('users', 'user_id', 'profiles', 'user_id', 'one_to_one'),
            ('companies', 'company_id', 'employees', 'company_id', 'one_to_many'),
            ('departments', 'department_id', 'employees', 'department_id', 'one_to_many'),
        ]
        
        # Get available collections and fields
        available_collections = set(collections_data.keys())
        
        for source_coll, source_field, target_coll, target_field, cardinality in patterns:
            if source_coll in available_collections and target_coll in available_collections:
                # Check if both fields exist
                source_has_field = self._collection_has_field(collections_data, source_coll, source_field)
                target_has_field = self._collection_has_field(collections_data, target_coll, target_field)
                
                if source_has_field and target_has_field:
                    relationships.append({
                        'source_collection': source_coll,
                        'source_field': source_field,
                        'target_collection': target_coll,
                        'target_field': target_field,
                        'cardinality': cardinality,
                        'join_type': 'left',
                        'confidence': 'medium',
                        'reason': f"Common pattern: {source_coll}({source_field}) → {target_coll}({target_field})",
                        'suggested_alias': target_coll
                    })
        
        return relationships

    def _generate_business_logic_relationships(self, collections_data: Dict[str, Any]) -> List[Dict]:
        """Generate relationships based on business logic and naming conventions."""
        relationships = []
        collections = list(collections_data.keys())
        
        # Business logic patterns
        business_patterns = [
            # User-centric relationships
            (['users', 'customers', 'clients'], ['orders', 'purchases', 'transactions']),
            (['users', 'customers'], ['profiles', 'accounts']),
            (['users', 'customers'], ['addresses', 'contacts']),
            
            # Product/Service relationships  
            (['products', 'items', 'services'], ['reviews', 'ratings', 'feedback']),
            (['products', 'items'], ['categories', 'types', 'groups']),
            (['orders', 'purchases'], ['order_items', 'line_items', 'details']),
            
            # Healthcare relationships
            (['patients'], ['appointments', 'visits', 'consultations']),
            (['patients'], ['treatments', 'procedures', 'therapies']),
            (['patients'], ['billing', 'invoices', 'payments']),
            (['doctors', 'physicians'], ['appointments', 'consultations']),
            (['doctors'], ['schedules', 'availability']),
            
            # Organizational relationships
            (['companies', 'organizations'], ['employees', 'staff', 'workers']),
            (['departments'], ['employees', 'staff']),
            (['projects'], ['tasks', 'activities', 'assignments']),
        ]
        
        for source_types, target_types in business_patterns:
            for source_type in source_types:
                for target_type in target_types:
                    if source_type in collections and target_type in collections:
                        # Find appropriate ID fields
                        source_id_field = f"{source_type[:-1]}_id"  # Remove 's' and add '_id'
                        target_id_field = source_id_field  # Same field name in target
                        
                        if (self._collection_has_field(collections_data, source_type, source_id_field) and
                            self._collection_has_field(collections_data, target_type, target_id_field)):
                            
                            relationships.append({
                                'source_collection': source_type,
                                'source_field': source_id_field,
                                'target_collection': target_type,
                                'target_field': target_id_field,
                                'cardinality': 'one_to_many',
                                'join_type': 'left',
                                'confidence': 'medium',
                                'reason': f"Business logic: {source_type} has many {target_type}",
                                'suggested_alias': target_type
                            })
        
        return relationships

    def _collection_has_field(self, collections_data: Dict[str, Any], collection: str, field: str) -> bool:
        """Check if a collection has a specific field."""
        if collection not in collections_data:
            return False
        
        collection_info = collections_data[collection]
        if isinstance(collection_info, dict):
            fields = collection_info.get('fields', [])
        else:
            fields = collection_info if isinstance(collection_info, list) else []
        
        field_names = [f['name'] for f in fields]
        return field in field_names

    def _remove_duplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships while preserving unique ones."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a unique key for this relationship
            key = (
                rel['source_collection'],
                rel['source_field'], 
                rel['target_collection'],
                rel['target_field'],
                rel['cardinality']
            )
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships

def get_ai_relationship_suggestions(collections_data: Dict[str, Any]) -> List[Dict]:
    """
    Convenience function to get comprehensive AI relationship suggestions.
    
    Args:
        collections_data: Dictionary with collection names as keys and field metadata as values
                         Can be either format:
                         - {"collection": [field_dicts...]} (old format)
                         - {"collection": {"description": "...", "fields": [field_dicts...]}} (new format)
        
    Returns:
        List of comprehensive relationship suggestions using multiple analysis strategies
    """
    try:
        suggester = AIRelationshipSuggester()
        
        # Use exhaustive analysis for maximum coverage
        return suggester.suggest_exhaustive_relationships(collections_data)
        
    except Exception as e:
        print(f"Failed to get AI suggestions: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic analysis
        try:
            suggester = AIRelationshipSuggester()
            return suggester.analyze_collections(collections_data)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
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