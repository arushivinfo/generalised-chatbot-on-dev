#!/usr/bin/env python3
"""
Test file to demonstrate Core Rules & Prompts functionality

This script shows how the Core Rules & Prompts section in the admin portal works:
1. How core rules are stored and retrieved
2. How different modes (auto, append, override) work
3. How extra context is managed
4. How the final prompt is constructed for the search agent
"""

import json
from typing import Dict, List
from schema_registry import (
    load_registry, save_registry,
    get_core_rules_config, set_core_rules_config,
    get_user_match_context, set_user_match_context,
    get_collection_names, get_all_fields, get_descriptions
)
from core_rules import render_core_rules, render_match_context, render_schema_section_all

def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_core_rules_modes():
    """Test the three core rules modes: auto, append, override"""
    
    print_separator("TESTING CORE RULES MODES")
    
    # Load current registry
    reg = load_registry()
    collection_names = get_collection_names(reg)
    
    print(f"Available collections: {collection_names}")
    
    # Test 1: AUTO mode (default generated rules)
    print("\n1. AUTO MODE (Generated Rules)")
    print("-" * 40)
    set_core_rules_config("auto", "")
    auto_rules = render_core_rules(collection_names)
    print("Auto-generated rules:")
    print(auto_rules)
    
    # Test 2: APPEND mode (generated + custom)
    print("\n2. APPEND MODE (Generated + Custom)")
    print("-" * 40)
    custom_text = """
**Additional Custom Rules:**
- Always prefer exact matches over fuzzy searches
- When dealing with dates, default to the last 30 days if no range specified
- For user queries about "recent" data, interpret as last 7 days
- Include confidence scores in responses when possible
"""
    set_core_rules_config("append", custom_text)
    
    # In append mode, the effective rules are auto + custom
    cfg = get_core_rules_config()
    if cfg["mode"] == "append" and cfg["custom_text"].strip():
        effective_rules = auto_rules + "\n\n" + cfg["custom_text"].strip()
    else:
        effective_rules = auto_rules
    
    print("Effective rules in APPEND mode:")
    print(effective_rules)
    
    # Test 3: OVERRIDE mode (only custom)
    print("\n3. OVERRIDE MODE (Custom Only)")
    print("-" * 40)
    override_text = """
**Custom Query Processing Rules:**

You are a specialized database query assistant. Follow these rules strictly:

1. **Query Optimization:**
   - Always use indexes when available
   - Limit results to maximum 100 items unless explicitly requested
   - Use aggregation pipelines for complex analytics

2. **Data Interpretation:**
   - Treat missing values as NULL, not empty strings
   - Convert all dates to ISO format
   - Handle case-insensitive string matching by default

3. **Response Format:**
   - Always include query execution time
   - Provide data source information
   - Include row counts in results

4. **Error Handling:**
   - Return descriptive error messages
   - Suggest alternative queries when original fails
   - Log all failed queries for analysis

These custom rules override all default behavior.
"""
    set_core_rules_config("override", override_text)
    
    cfg = get_core_rules_config()
    if cfg["mode"] == "override" and cfg["custom_text"].strip():
        effective_rules = cfg["custom_text"].strip()
    else:
        effective_rules = auto_rules
    
    print("Effective rules in OVERRIDE mode:")
    print(effective_rules)

def test_user_match_context():
    """Test the extra context functionality"""
    
    print_separator("TESTING USER MATCH CONTEXT (EXTRA CONTEXT)")
    
    # Test setting and getting extra context
    extra_context = """
**Special Instructions for this deployment:**

- This system is deployed for a healthcare organization
- Patient privacy is paramount - always verify user permissions
- Financial data requires additional authentication
- Emergency queries should be prioritized
- All queries are logged for compliance auditing

**Business Context:**
- Fiscal year runs April to March
- Peak hours are 9 AM - 5 PM EST
- System maintenance window is Sunday 2-4 AM EST
"""
    
    print("Setting extra context...")
    set_user_match_context(extra_context)
    
    print("\nRetrieving extra context:")
    retrieved_context = get_user_match_context()
    print(retrieved_context)
    
    print("\nRendered match context:")
    rendered = render_match_context(retrieved_context)
    print(rendered)

def test_complete_prompt_construction():
    """Test how the complete prompt is constructed for the search agent"""
    
    print_separator("TESTING COMPLETE PROMPT CONSTRUCTION")
    
    # Get current configuration
    reg = load_registry()
    core_rules_cfg = get_core_rules_config(reg)
    user_context = get_user_match_context(reg)
    
    # Get schema information
    collection_names = get_collection_names(reg)
    all_fields = get_all_fields(reg)
    descriptions = get_descriptions(reg)
    options_max = reg.get("options_max", 20)
    
    print("1. SCHEMA SECTION:")
    print("-" * 20)
    schema_section = render_schema_section_all(all_fields, descriptions, options_max)
    print(schema_section)
    
    print("\n2. CORE RULES SECTION:")
    print("-" * 20)
    auto_rules = render_core_rules(collection_names)
    
    # Apply the current mode
    if core_rules_cfg["mode"] == "override" and core_rules_cfg["custom_text"].strip():
        effective_rules = core_rules_cfg["custom_text"].strip()
    elif core_rules_cfg["mode"] == "append" and core_rules_cfg["custom_text"].strip():
        effective_rules = auto_rules + "\n\n" + core_rules_cfg["custom_text"].strip()
    else:
        effective_rules = auto_rules
    
    print(effective_rules)
    
    print("\n3. USER MATCH CONTEXT:")
    print("-" * 20)
    print(user_context or "(empty)")
    
    print("\n4. COMPLETE SYSTEM PROMPT (as sent to LLM):")
    print("-" * 20)
    
    # This simulates what happens in search_agent_new.py
    base_system_prompt = f"""You are an expert MongoDB query planner for the following collections:

{schema_section}

Your task is to convert natural language queries into precise MongoDB aggregation pipelines."""
    
    complete_prompt = f"""{base_system_prompt}

{effective_rules}

{user_context}
"""
    
    print(complete_prompt)

def test_configuration_persistence():
    """Test that configuration changes persist across restarts"""
    
    print_separator("TESTING CONFIGURATION PERSISTENCE")
    
    # Save current state
    original_rules = get_core_rules_config()
    original_context = get_user_match_context()
    
    print("Original configuration:")
    print(f"Rules mode: {original_rules['mode']}")
    print(f"Custom text length: {len(original_rules['custom_text'])} chars")
    print(f"Context length: {len(original_context)} chars")
    
    # Make changes
    test_rules = "Test custom rules for persistence check"
    test_context = "Test context for persistence check"
    
    set_core_rules_config("append", test_rules)
    set_user_match_context(test_context)
    
    print("\nAfter making changes:")
    new_rules = get_core_rules_config()
    new_context = get_user_match_context()
    print(f"Rules mode: {new_rules['mode']}")
    print(f"Custom text: {new_rules['custom_text']}")
    print(f"Context: {new_context}")
    
    # Verify persistence by reloading from disk
    print("\nVerifying persistence (reloading from disk):")
    reloaded_rules = get_core_rules_config()
    reloaded_context = get_user_match_context()
    
    assert reloaded_rules['mode'] == 'append'
    assert reloaded_rules['custom_text'] == test_rules
    assert reloaded_context == test_context
    print("✅ Configuration persisted correctly!")
    
    # Restore original state
    set_core_rules_config(original_rules['mode'], original_rules['custom_text'])
    set_user_match_context(original_context)
    print("\n✅ Original configuration restored")

def demo_admin_workflow():
    """Demonstrate a typical admin workflow using the Core Rules & Prompts"""
    
    print_separator("DEMO: TYPICAL ADMIN WORKFLOW")
    
    print("Scenario: Admin wants to customize the chatbot for a specific business domain")
    print("\nStep 1: Check current configuration")
    print("-" * 40)
    
    current_rules = get_core_rules_config()
    current_context = get_user_match_context()
    
    print(f"Current mode: {current_rules['mode']}")
    print(f"Has custom rules: {'Yes' if current_rules['custom_text'].strip() else 'No'}")
    print(f"Has extra context: {'Yes' if current_context.strip() else 'No'}")
    
    print("\nStep 2: Add business-specific rules (APPEND mode)")
    print("-" * 40)
    
    business_rules = """
**E-commerce Business Rules:**

- Product queries should include stock status by default
- Order queries should show order status and shipping info
- Customer queries should respect privacy settings
- Price queries should include tax and shipping calculations
- Inventory queries should flag low-stock items (< 10 units)
- Returns/refunds require special handling with approval workflow
"""
    
    set_core_rules_config("append", business_rules)
    print("✅ Business rules added in APPEND mode")
    
    print("\nStep 3: Add deployment-specific context")
    print("-" * 40)
    
    deployment_context = """
**Deployment Context:**

Environment: Production E-commerce Platform
Region: North America (EST timezone)
Business Hours: Mon-Fri 9 AM - 6 PM EST
Peak Season: November-January (holiday shopping)
Currency: USD
Tax Calculation: Varies by state
Shipping: 2-day default, expedited available
Customer Service: Available during business hours
"""
    
    set_user_match_context(deployment_context)
    print("✅ Deployment context added")
    
    print("\nStep 4: Preview final configuration")
    print("-" * 40)
    
    # Show how it would appear to the LLM
    reg = load_registry()
    collection_names = get_collection_names(reg)
    auto_rules = render_core_rules(collection_names)
    final_rules = auto_rules + "\n\n" + business_rules
    
    print("Final effective rules:")
    print(final_rules)
    print("\nFinal context:")
    print(deployment_context)
    
    print("\n✅ Configuration ready for production use!")

if __name__ == "__main__":
    print("🧪 Core Rules & Prompts Functionality Test")
    print("This script demonstrates how the admin portal's Core Rules & Prompts section works")
    
    try:
        # Run all tests
        test_core_rules_modes()
        test_user_match_context()
        test_complete_prompt_construction()
        test_configuration_persistence()
        demo_admin_workflow()
        
        print_separator("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
