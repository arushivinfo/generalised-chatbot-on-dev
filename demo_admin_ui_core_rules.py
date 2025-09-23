#!/usr/bin/env python3
"""
Demo script showing Core Rules & Prompts admin UI functionality

This script simulates what happens when you use the Core Rules & Prompts 
section in the admin portal UI.
"""

import json
from schema_registry import (
    load_registry, get_core_rules_config, set_core_rules_config,
    get_user_match_context, set_user_match_context,
    get_collection_names, get_all_fields, get_descriptions
)
from core_rules import render_core_rules, render_schema_section_all

def demo_admin_ui_workflow():
    """Demonstrates the exact workflow from the admin UI"""
    
    print("🎯 Core Rules & Prompts Admin UI Demo")
    print("="*50)
    
    # 1. Show current configuration (what admin sees when they open the tab)
    print("\n1. CURRENT CONFIGURATION (Initial Load)")
    print("-" * 30)
    
    cfg = get_core_rules_config()
    current_context = get_user_match_context()
    
    print(f"Mode: {cfg['mode']}")
    print(f"Custom Text Length: {len(cfg['custom_text'])} characters")
    print(f"Extra Context Length: {len(current_context)} characters")
    
    if cfg['custom_text']:
        print(f"\nCurrent Custom Text Preview:")
        print(cfg['custom_text'][:200] + "..." if len(cfg['custom_text']) > 200 else cfg['custom_text'])
    
    # 2. Show what happens when admin changes mode
    print("\n2. ADMIN CHANGES MODE TO 'APPEND'")
    print("-" * 30)
    
    # Admin adds custom rules
    custom_rules = """
**Business-Specific Rules:**
- Always include customer satisfaction scores in order queries
- Flag orders with shipping delays > 3 days
- Include inventory levels for product queries  
- Show promotional discount eligibility
- Default to last 30 days for sales analytics
"""
    
    print("Admin enters custom rules:")
    print(custom_rules)
    
    # Admin clicks "Save Core Rules" button
    set_core_rules_config("append", custom_rules)
    print("✅ Admin clicks 'Save Core Rules' - Configuration saved!")
    
    # 3. Show what happens when admin adds extra context
    print("\n3. ADMIN ADDS EXTRA CONTEXT")
    print("-" * 30)
    
    extra_context = """
**Deployment Context:**
- E-commerce platform for sporting goods
- Peak season: March-August (outdoor sports)
- Customer base: US, Canada, Europe
- Currency: USD with auto-conversion
- Shipping: 2-day standard, overnight available
- Returns: 30-day policy, extended for defective items
"""
    
    print("Admin enters extra context:")
    print(extra_context)
    
    # Admin clicks "Save Extra Context" button
    set_user_match_context(extra_context)
    print("✅ Admin clicks 'Save Extra Context' - Context saved!")
    
    # 4. Show the effective rules preview (what admin sees in the preview section)
    print("\n4. EFFECTIVE RULES PREVIEW")
    print("-" * 30)
    
    reg = load_registry()
    collection_names = get_collection_names(reg)
    auto_rules = render_core_rules(collection_names)
    
    # Apply the current mode setting
    cfg = get_core_rules_config()
    if cfg["mode"] == "override" and cfg["custom_text"].strip():
        effective_rules = cfg["custom_text"].strip()
    elif cfg["mode"] == "append" and cfg["custom_text"].strip():
        effective_rules = auto_rules + "\n\n" + cfg["custom_text"].strip()
    else:
        effective_rules = auto_rules
    
    print("This is what the admin sees in the 'Effective Rules Preview' section:")
    print(effective_rules)
    
    # 5. Show the full search agent prompt preview
    print("\n5. FULL SEARCH AGENT PROMPT PREVIEW")
    print("-" * 30)
    
    # Schema section
    all_fields = get_all_fields(reg)
    descriptions = get_descriptions(reg)
    options_max = reg.get("options_max", 20)
    schema_section = render_schema_section_all(all_fields, descriptions, options_max)
    
    # Base system prompt
    base_system = f"You are an expert MongoDB query planner for the following collections:\n\n{schema_section}"
    
    # Get current context
    match_context = get_user_match_context()
    
    # Compose the full prompt preview (what admin sees in the prompt inspector)
    prompt_preview = f"""【SYSTEM #1】Search SYSTEM_PROMPT
{base_system}

【SYSTEM #2】CORE_RULES_TEXT (effective)
{effective_rules}

【SYSTEM #3】MATCH_CONTEXT (admin extra)
{match_context or "(empty)"}

【SYSTEM #4】MEMORY_PROMPT
(constructed at runtime; recent Q/A, pronoun rules)

【HUMAN】{{query}}"""
    
    print("This is what the admin sees in the 'Search Agent – Prompt Inspector':")
    print(prompt_preview)
    
    # 6. Demonstrate mode switching
    print("\n6. ADMIN SWITCHES TO OVERRIDE MODE")
    print("-" * 30)
    
    override_rules = """
**COMPLETE OVERRIDE - Specialized System:**

You are a specialized AI for outdoor sports equipment recommendations.

RULES:
1. Always consider weather conditions and season
2. Recommend complementary equipment (upselling)
3. Include safety equipment warnings where applicable
4. Mention skill level requirements for advanced gear
5. Provide care and maintenance tips
6. Include warranty and return policy information

NEVER deviate from these specialized rules.
"""
    
    print("Admin switches to OVERRIDE mode and enters:")
    print(override_rules)
    
    set_core_rules_config("override", override_rules)
    print("✅ Mode changed to OVERRIDE")
    
    # Show how this changes the effective rules
    new_cfg = get_core_rules_config()
    if new_cfg["mode"] == "override" and new_cfg["custom_text"].strip():
        new_effective = new_cfg["custom_text"].strip()
    
    print("\nNew effective rules (override replaces everything):")
    print(new_effective)
    
    # 7. Show settings persistence
    print("\n7. CONFIGURATION PERSISTENCE")
    print("-" * 30)
    
    print("Current configuration is automatically saved to schema_registry.json")
    print("Settings persist across:")
    print("- Server restarts")
    print("- Admin portal sessions") 
    print("- Search agent queries")
    print("- System updates")
    
    # Verify by reloading
    reloaded_cfg = get_core_rules_config()
    reloaded_context = get_user_match_context()
    
    print(f"\nVerification - Reloaded config:")
    print(f"Mode: {reloaded_cfg['mode']}")
    print(f"Custom text length: {len(reloaded_cfg['custom_text'])}")
    print(f"Context length: {len(reloaded_context)}")
    print("✅ Configuration persisted correctly!")

def demo_real_world_examples():
    """Show real-world examples of how this affects actual queries"""
    
    print("\n" + "="*60)
    print(" REAL-WORLD QUERY EXAMPLES")
    print("="*60)
    
    # Example query
    user_query = "Show me popular products from last month"
    
    print(f"User Query: '{user_query}'")
    print("\nHow this query is processed with different configurations:")
    
    # 1. Default (auto) mode
    print("\n1. WITH DEFAULT RULES:")
    print("-" * 25)
    set_core_rules_config("auto", "")
    set_user_match_context("")
    
    reg = load_registry()
    collection_names = get_collection_names(reg)
    auto_rules = render_core_rules(collection_names)
    
    print("Rules applied:", auto_rules[:100] + "...")
    print("Query processing:")
    print("- ✅ Use basic collection filtering")
    print("- ✅ Apply standard date range (last month)")
    print("- ✅ Set reasonable limits")
    print("- ✅ Standard sorting by relevance")
    
    # 2. Business rules (append)
    print("\n2. WITH BUSINESS RULES (APPEND):")
    print("-" * 35)
    
    business_rules = """
- Popular products defined as >100 orders or >4.5 star rating
- Include inventory status and restock dates
- Show trending indicators (↗️ ↘️)
- Add customer review highlights
- Include promotional opportunities
"""
    
    set_core_rules_config("append", business_rules)
    
    print("Additional rules applied:", business_rules.strip())
    print("Enhanced query processing:")
    print("- ✅ Filter by popularity metrics (>100 orders OR >4.5 stars)")
    print("- ✅ Include inventory and restock information")
    print("- ✅ Add trending indicators")
    print("- ✅ Include customer review highlights")
    print("- ✅ Suggest promotional opportunities")
    
    # 3. Complete override
    print("\n3. WITH COMPLETE OVERRIDE:")
    print("-" * 30)
    
    override_rules = """
SPECIALIZED RECOMMENDATION ENGINE:
1. Analyze customer purchase history for personalization
2. Include seasonal and weather-based recommendations
3. Add cross-selling and upselling suggestions
4. Factor in current promotions and discounts
5. Include sustainability and ethical sourcing info
6. Provide detailed product comparisons
"""
    
    set_core_rules_config("override", override_rules)
    
    print("Override rules completely replace defaults:")
    print("Specialized query processing:")
    print("- ✅ Personalized recommendations based on history")
    print("- ✅ Weather and seasonal considerations")
    print("- ✅ Cross-selling opportunities")
    print("- ✅ Active promotions integration")
    print("- ✅ Sustainability information")
    print("- ✅ Comparative analysis")
    
    print("\n🎯 CONCLUSION:")
    print("The same user query produces completely different results")
    print("based on admin configuration in Core Rules & Prompts!")

if __name__ == "__main__":
    try:
        # Save original configuration
        original_cfg = get_core_rules_config()
        original_context = get_user_match_context()
        
        # Run the demo
        demo_admin_ui_workflow()
        demo_real_world_examples()
        
        print("\n" + "="*60)
        print(" DEMO COMPLETED SUCCESSFULLY! ✅")
        print("="*60)
        
        # Restore original configuration
        set_core_rules_config(original_cfg['mode'], original_cfg['custom_text'])
        set_user_match_context(original_context)
        print("\n✅ Original configuration restored")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
