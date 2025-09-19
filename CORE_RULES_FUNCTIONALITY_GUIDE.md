# Core Rules & Prompts Functionality Guide

## Overview

The "Core Rules & Prompts" section in the admin portal is a powerful feature that allows administrators to customize how the AI chatbot interprets and responds to user queries. This system controls the core behavior and prompt engineering for the search agent.

## How It Works

### 1. Architecture

```
User Query → Search Agent → LLM
                ↑
        Core Rules & Prompts
        (Admin Configuration)
```

The Core Rules & Prompts configuration is injected into the system prompt sent to the LLM, fundamentally changing how it processes queries.

### 2. Three Operating Modes

#### AUTO Mode (Default)
- **Purpose**: Provides sensible, auto-generated rules based on your collections
- **Behavior**: System generates standard MongoDB query guidance
- **Use Case**: When you want basic, reliable query processing without customization

**Example Auto Rules:**
```
**Guidance for answering**
- Use the smallest set of collections needed; prefer structured filters (field:value, ranges)
- Always set sort and limit explicitly when the intent implies ordering or top-K
- If a question mixes "current status/forecast" with history, query both collections
- Available collections include: patients, appointments, treatments
```

#### APPEND Mode (Recommended)
- **Purpose**: Combines auto-generated rules with custom business logic
- **Behavior**: Auto rules + your custom text
- **Use Case**: When you want to add business-specific rules while keeping defaults

**Example with Business Rules:**
```
[Auto-generated rules above]

**Healthcare-Specific Rules:**
- Patient data queries must include privacy compliance checks
- Medical records require authorization verification
- Date ranges for medical history default to last 2 years unless specified
- All queries must be HIPAA-compliant
```

#### OVERRIDE Mode (Advanced)
- **Purpose**: Complete control over query processing behavior
- **Behavior**: Only your custom rules (ignores auto-generated)
- **Use Case**: Specialized systems with unique requirements

**Example Override:**
```
**Financial Trading System:**
You are an AI for real-time financial analytics.

CRITICAL REQUIREMENTS:
1. All queries must specify exact time ranges (no defaults)
2. Price data must include currency and exchange information
3. Risk calculations must include volatility measures
4. Market data must be real-time (< 5 seconds old)

These rules completely replace all default behavior.
```

### 3. Extra Context

In addition to core rules, you can add deployment-specific context that provides environmental information to the AI:

```
**Deployment Context:**
- Healthcare network with 5 hospitals, 12 clinics
- 24/7 emergency services
- Specialties: Cardiology, Oncology, Pediatrics
- HIPAA compliance required
- Audit trail mandatory for all data access
```

## Complete Prompt Structure

When a user asks a question, the search agent constructs this prompt hierarchy:

```
【SYSTEM #1】Base System Prompt
You are an expert MongoDB query planner for the following collections:
[Schema information with fields, types, and sample values]

【SYSTEM #2】Core Rules (from Admin)
[Auto-generated rules + custom rules based on mode]

【SYSTEM #3】Extra Context (from Admin)
[Deployment-specific context and instructions]

【SYSTEM #4】Memory & History
[Recent conversation context and pronoun resolution]

【HUMAN】User Query
[The actual user question]
```

## Real-World Examples

### E-commerce Platform

**Mode**: APPEND
**Custom Rules**:
```
**E-commerce Rules:**
- Product queries should include stock status by default
- Order queries should show order status and shipping info
- Price queries should include tax and shipping calculations
- Inventory queries should flag low-stock items (< 10 units)
```

**Extra Context**:
```
**Deployment:**
- E-commerce platform for sporting goods
- Peak season: March-August (outdoor sports)
- Currency: USD with auto-conversion
- Returns: 30-day policy
```

**Result**: User query "Show popular products" becomes sophisticated recommendation engine with inventory awareness, seasonal considerations, and business context.

### Healthcare System

**Mode**: APPEND
**Custom Rules**:
```
**Healthcare Rules:**
- Patient data queries must include privacy compliance checks
- Medical records require authorization verification
- Emergency cases take priority over routine requests
- Include doctor/provider information in patient queries
```

**Extra Context**:
```
**Healthcare Network:**
- Multi-location healthcare system
- HIPAA compliance mandatory
- Electronic Health Records integration
- Strict audit trail requirements
```

**Result**: User query "Find diabetes patients" becomes privacy-aware, compliant query with proper authorization checks and audit logging.

### Financial Trading Platform

**Mode**: OVERRIDE
**Custom Rules**:
```
**Trading Analytics System:**
You are an AI for real-time financial trading analytics.

CRITICAL REQUIREMENTS:
1. All queries must specify exact time ranges
2. Include currency and exchange information
3. Risk calculations must include volatility measures
4. Market data must be real-time (< 5 seconds old)
5. Performance calculations use time-weighted returns

RESPONSE FORMAT:
- Always include data source and timestamp
- Show calculation methodology
- Provide statistical significance levels
- Include risk warnings where applicable
```

**Result**: User query "Show tech stock performance" becomes sophisticated financial analysis with risk assessment, regulatory compliance, and real-time validation.

## Testing and Validation

### Test Files Included

1. **`test_core_rules_functionality.py`**
   - Comprehensive test of all functionality
   - Demonstrates mode switching
   - Shows configuration persistence
   - Tests prompt construction

2. **`test_core_rules_scenarios.py`**
   - Interactive scenarios for different industries
   - Security and prompt injection protection
   - Multi-environment configurations
   - Comparison analysis

3. **`demo_admin_ui_core_rules.py`**
   - Simulates exact admin UI workflow
   - Shows step-by-step configuration process
   - Demonstrates real-world query impact

### Running Tests

```bash
# Basic functionality test
python test_core_rules_functionality.py

# Interactive scenarios
python test_core_rules_scenarios.py

# Admin UI simulation
python demo_admin_ui_core_rules.py
```

## Best Practices

### 1. Start with APPEND Mode
- Begin with auto-generated rules
- Add business-specific customizations gradually
- Test thoroughly before moving to production

### 2. Use Clear, Specific Language
- Write rules as direct instructions to the AI
- Be specific about data requirements
- Include error handling expectations

### 3. Consider Security
- Include privacy and compliance requirements
- Add input validation rules
- Specify access control requirements

### 4. Test Different Scenarios
- Test with various user query types
- Verify business rules are applied correctly
- Check that security requirements are enforced

### 5. Environment-Specific Configuration
- Use different rules for dev/staging/production
- Include environment context in extra context
- Document configuration changes

## Configuration Persistence

All Core Rules & Prompts settings are automatically saved to `schema_registry.json` and persist across:
- Server restarts
- Admin portal sessions
- Search agent queries
- System updates

## Impact on Query Processing

The same user query produces dramatically different results based on configuration:

**Query**: "Show me recent customer orders"

**With Default Rules**:
- Basic order filtering
- Standard date range (recent = reasonable default)
- Default result limits

**With E-commerce Rules**:
- Orders filtered by business-defined "recent" (e.g., last 14 days)
- Include customer tier information
- Show order fulfillment status
- Add promotional opportunities

**With Financial Rules**:
- Exact time range validation required
- Currency conversion included
- Risk assessment scores
- Regulatory compliance markers

## Troubleshooting

### Rules Not Taking Effect
1. Check that configuration was saved (look for success message)
2. Verify mode is set correctly (auto/append/override)
3. Test with simple query to confirm behavior change

### Unexpected Behavior
1. Review effective rules in the preview section
2. Check for conflicting instructions
3. Verify extra context doesn't contradict core rules

### Performance Issues
1. Keep custom rules concise and clear
2. Avoid overly complex instructions
3. Test with representative queries

## Conclusion

The Core Rules & Prompts functionality is a powerful tool for customizing your AI chatbot's behavior. By understanding the three modes (AUTO, APPEND, OVERRIDE) and properly using extra context, you can create sophisticated, domain-specific query processing that meets your exact business requirements.

The test files provided demonstrate comprehensive functionality and can serve as templates for your own testing and validation processes.
