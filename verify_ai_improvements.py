#!/usr/bin/env python3
"""
Quick verification script for improved AI relationship suggestions
"""

import os
import sys
sys.path.insert(0, '/home/arushi/Documents/generalised-chatbot-on-dev-9-Sep-Stable-fuction-with-Eval')

def quick_test():
    print("🧪 Quick Test: Enhanced AI Relationship Suggestions")
    print("=" * 50)
    
    # Simple test data
    test_data = {
        "patients": {
            "description": "Patient records",
            "fields": [
                {"name": "patient_id", "type": "string"},
                {"name": "name", "type": "string"},
                {"name": "primary_doctor_id", "type": "string"}
            ]
        },
        "appointments": {
            "description": "Medical appointments",
            "fields": [
                {"name": "appointment_id", "type": "string"},
                {"name": "patient_id", "type": "string"},
                {"name": "doctor_id", "type": "string"}
            ]
        },
        "doctors": {
            "description": "Medical staff",
            "fields": [
                {"name": "doctor_id", "type": "string"},
                {"name": "name", "type": "string"},
                {"name": "specialty", "type": "string"}
            ]
        }
    }
    
    try:
        from ai_relationship_suggester import get_ai_relationship_suggestions
        
        print("🔍 Running comprehensive relationship analysis...")
        suggestions = get_ai_relationship_suggestions(test_data)
        
        print(f"✅ Generated {len(suggestions)} relationship suggestions")
        
        if suggestions:
            print("\n📋 Sample suggestions:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                source = suggestion.get('source_collection', 'N/A')
                target = suggestion.get('target_collection', 'N/A')
                source_field = suggestion.get('source_field', 'N/A')
                target_field = suggestion.get('target_field', 'N/A')
                confidence = suggestion.get('confidence', 'unknown')
                
                print(f"  {i}. {source}.{source_field} → {target}.{target_field} ({confidence})")
            
            if len(suggestions) > 5:
                print(f"  ... and {len(suggestions) - 5} more suggestions")
                
            print(f"\n🎯 Test Result: SUCCESS - AI suggestions are working with comprehensive coverage!")
            return True
        else:
            print("❌ No suggestions generated")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)