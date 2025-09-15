# llm_services.py - Simple LLM service management for Perfect Lineup Cricket Chatbot

import os
import time
import logging
import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

# Configure LiteLLM
litellm.set_verbose = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration file path
MODEL_CONFIG_FILE = "model_config.json"

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "search_agent_model": "gpt-4o-mini",
    "narrator_model": "gpt-4o-mini",
    "admin_ai_model": "gpt-4o-mini",
    "last_updated": None,
    "version": "1.0"
}

def load_model_config() -> Dict[str, Any]:
    """Load model configuration from JSON file"""
    try:
        if os.path.exists(MODEL_CONFIG_FILE):
            with open(MODEL_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # logger.info(f"[MODEL_CONFIG] Loaded from {MODEL_CONFIG_FILE}: {config}")
                return config
        else:
            # logger.info(f"[MODEL_CONFIG] File {MODEL_CONFIG_FILE} not found, using defaults")
            return DEFAULT_MODEL_CONFIG.copy()
    except Exception as e:
        logger.error(f"[MODEL_CONFIG] Error loading config: {e}")
        return DEFAULT_MODEL_CONFIG.copy()

def save_model_config(config: Dict[str, Any]) -> bool:
    """Save model configuration to JSON file"""
    try:
        config["last_updated"] = datetime.now().isoformat()
        with open(MODEL_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"[MODEL_CONFIG] Saved to {MODEL_CONFIG_FILE}: {config}")
        return True
    except Exception as e:
        logger.error(f"[MODEL_CONFIG] Error saving config: {e}")
        return False

def get_model_for_role(role: str) -> str:
    """Get model for role directly from JSON config file"""
    config = load_model_config()
    key = f"{role}_model"
    model = config.get(key, "gpt-4o-mini")
    # logger.info(f"[MODEL_CONFIG] Role '{role}' using model: {model}")
    return model

def update_model_for_role(role: str, model_id: str) -> bool:
    """Update model for role in JSON config file"""
    config = load_model_config()
    key = f"{role}_model"
    old_model = config.get(key, "unknown")
    config[key] = model_id
    success = save_model_config(config)
    if success:
        logger.info(f"[MODEL_CONFIG] Updated {role}: {old_model} → {model_id}")
    return success

def setup_api_keys():
    """Configure API keys for different providers"""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") 
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    # Configure LiteLLM with API keys
    if openai_key:
        litellm.api_key = openai_key
        os.environ["OPENAI_API_KEY"] = openai_key
        
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get available models grouped by provider"""
    # Check which providers are available
    providers_available = {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "Groq": bool(os.getenv("GROQ_API_KEY")),
    }
    
    # Define all models
    all_models = {
        "OpenAI": [
            {"name": "GPT-4.1 Mini", "id": "gpt-4.1-mini", "cost": "Low", "speed": "Fast"},
            {"name": "GPT-4o Mini", "id": "gpt-4o-mini", "cost": "Low", "speed": "Fast"},
            {"name": "GPT-4o", "id": "gpt-4o", "cost": "High", "speed": "Medium"},
            {"name": "GPT-3.5 Turbo", "id": "gpt-3.5-turbo", "cost": "Very Low", "speed": "Very Fast"},
        ],
        "Anthropic": [
            {"name": "Claude 3.5 Sonnet", "id": "claude-3-5-sonnet-20241022", "cost": "High", "speed": "Medium"},
            {"name": "Claude 3.5 Haiku", "id": "claude-3-5-haiku-20241022", "cost": "Low", "speed": "Fast"},
        ],
        "Google": [
            {"name": "Gemini 1.5 Pro", "id": "gemini/gemini-1.5-pro", "cost": "Medium", "speed": "Medium"},
            {"name": "Gemini 1.5 Flash", "id": "gemini/gemini-1.5-flash", "cost": "Low", "speed": "Fast"},
        ],
        "Groq": [
            {"name": "Llama 3.1 70B", "id": "groq/llama-3.1-70b-versatile", "cost": "Free", "speed": "Very Fast"},
            {"name": "Mixtral 8x7B", "id": "groq/mixtral-8x7b-32768", "cost": "Free", "speed": "Very Fast"},
        ],
    }
    
    # Filter to only available providers
    return {provider: models for provider, models in all_models.items() 
            if providers_available.get(provider, False)}

def call_llm(messages: List[Dict[str, str]], role: str, **kwargs):
    """Universal LLM call that reads model from JSON config"""
    setup_api_keys()
    
    # Get model from JSON config
    model_id = get_model_for_role(role)
    # logger.info(f"[LLM_CALL] Using model for {role}: {model_id}")
    
    params = {
        "model": model_id,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.1),
        "max_tokens": kwargs.get("max_tokens", 4096),
        "stream": kwargs.get("stream", False)
    }
    
    try:
        start = time.time()
        response = litellm.completion(**params)
        # logger.info(f"[LLM_CALL] {role} ({model_id}) took {time.time() - start:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"[LLM_CALL] {role} ({model_id}) failed: {e}")
        # Fallback to GPT-3.5 if available
        if model_id != "gpt-3.5-turbo":
            params["model"] = "gpt-3.5-turbo"
            return litellm.completion(**params)
        raise

def call_narrator_model(messages: List[Dict[str, str]], stream: bool, **kwargs):
    """Call narrator model - supports streaming for response generation"""
    model_id = get_model_for_role("narrator")
    print(f"[NARRATOR_CALL] Reading from JSON config - using model: {model_id}")

    if stream:
        # Streaming version -> generator
        kwargs['stream'] = True
        response = call_llm(messages, role="narrator", **kwargs)

        def stream_generator():
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        return stream_generator()   # returns generator
    else:
        # Non-streaming version -> normal text
        response = call_llm(messages, role="narrator", **kwargs)
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        return str(response)

def call_admin_ai_model(messages: List[Dict[str, str]], **kwargs):
    """Call the admin AI model - returns text content only"""
    model_id = get_model_for_role("admin_ai")
    print(f"[ADMIN_AI_CALL] Reading from JSON config - using model: {model_id}")
    
    response = call_llm(messages, role="admin_ai", **kwargs)
    
    # Extract text content from LiteLLM response
    if hasattr(response, 'choices') and len(response.choices) > 0:
        return response.choices[0].message.content.strip()
    else:
        return str(response)

def call_search_agent_model(messages: List[Dict[str, str]], **kwargs):
    """Call the search agent model - returns text content only"""
    # Debug: Show which model is being used
    model_id = get_model_for_role("search_agent")
    print(f"[SEARCH_AGENT_CALL] Reading from JSON config - using model: {model_id}")
    # logger.info(f"[SEARCH_AGENT_CALL] Reading from JSON config - using model: {model_id}")
    
    response = call_llm(messages, role="search_agent", **kwargs)
    
    # Extract text content from LiteLLM response
    if hasattr(response, 'choices') and len(response.choices) > 0:
        return response.choices[0].message.content.strip()
    else:
        return str(response)

def render_llm_model_config():
    """Simple admin interface for LLM model configuration"""
    st.header("🔧 LLM Model Configuration")
    
    # Show current configuration
    current_config = load_model_config()
    
    st.subheader("📋 Current Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Search Agent Model:**")
        st.code(current_config.get("search_agent_model", "Not Set"))
        
    with col2:
        st.write("**Narrator Model:**")  
        st.code(current_config.get("narrator_model", "Not Set"))

    with col3:
        st.write("**Admin AI Model:**")  
        st.code(current_config.get("admin_ai_model", "Not Set"))
    
    # Show available providers
    st.subheader("🔑 API Key Status")
    key_status_cols = st.columns(4)
    
    with key_status_cols[0]:
        openai_status = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
        st.write(f"{openai_status} OpenAI")
    
    with key_status_cols[1]:
        anthropic_status = "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌"
        st.write(f"{anthropic_status} Anthropic")
    
    with key_status_cols[2]:
        google_status = "✅" if os.getenv("GOOGLE_API_KEY") else "❌"
        st.write(f"{google_status} Google")
    
    with key_status_cols[3]:
        groq_status = "✅" if os.getenv("GROQ_API_KEY") else "❌"
        st.write(f"{groq_status} Groq")
    
    # Show available providers
    st.subheader("📡 Available Providers")
    available_models = get_available_models()

    if not available_models:
        st.error("No API keys configured. Please set environment variables.")
        st.info("Required: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY")
        return

    # Display available providers and their models
    for provider, models in available_models.items():
        with st.expander(f"{provider} ({len(models)} models)", expanded=False):
            for model in models:
                st.write(f"• **{model['name']}** (`{model['id']}`) - Cost: {model['cost']}, Speed: {model['speed']}")

    # Create model options
    all_model_options = []
    for provider, models in available_models.items():
        for model in models:
            display_name = f"{model['name']} ({provider}) - {model['cost']}, {model['speed']}"
            all_model_options.append((display_name, model['id']))
    st.subheader("🎯 Select Models")
    
    # Model selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🔍 Search Agent Model**")
        st.caption("Handles query planning and database searches")
        
        # Find current index
        current_search_model = current_config.get("search_agent_model", "gpt-4o-mini")
        search_index = 0
        for idx, (_, model_id) in enumerate(all_model_options):
            if model_id == current_search_model:
                search_index = idx
                break
        
        search_selection = st.selectbox(
            "Search Agent Model",
            options=[option[0] for option in all_model_options],
            index=search_index,
            key="search_agent_select",
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("**📝 Narrator Model**")
        st.caption("Generates final cricket analysis responses")
        
        # Find current index
        current_narrator_model = current_config.get("narrator_model", "gpt-4o-mini") 
        narrator_index = 0
        for idx, (_, model_id) in enumerate(all_model_options):
            if model_id == current_narrator_model:
                narrator_index = idx
                break
        
        narrator_selection = st.selectbox(
            "Narrator Model", 
            options=[option[0] for option in all_model_options],
            index=narrator_index,
            key="narrator_select",
            label_visibility="collapsed"
        )

    with col3:
        st.write("**🤖 Admin AI Model**")
        st.caption("Handles administrative tasks")
        
        # Find current index
        current_admin_ai_model = current_config.get("admin_ai_model", "gpt-4o-mini") 
        admin_ai_index = 0
        for idx, (_, model_id) in enumerate(all_model_options):
            if model_id == current_admin_ai_model:
                admin_ai_index = idx
                break
        
        admin_ai_selection = st.selectbox(
            "Admin AI Model", 
            options=[option[0] for option in all_model_options],
            index=admin_ai_index,
            key="admin_ai_select",
            label_visibility="collapsed"
        )
    
    # Apply changes button
    st.subheader("💾 Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Apply Changes", type="primary"):
            # Get selected model IDs
            search_model_id = None
            narrator_model_id = None
            admin_ai_model_id = None
            
            for display_name, model_id in all_model_options:
                if display_name == search_selection:
                    search_model_id = model_id
                if display_name == narrator_selection:
                    narrator_model_id = model_id
                if display_name == admin_ai_selection:
                    admin_ai_model_id = model_id
            
            # Update configuration
            changes_made = False
            
            if search_model_id and search_model_id != current_search_model:
                update_model_for_role("search_agent", search_model_id)
                st.success(f"✅ Search Agent: {current_search_model} → {search_model_id}")
                changes_made = True
            
            if narrator_model_id and narrator_model_id != current_narrator_model:
                update_model_for_role("narrator", narrator_model_id) 
                st.success(f"✅ Narrator: {current_narrator_model} → {narrator_model_id}")
                changes_made = True

            if admin_ai_model_id and admin_ai_model_id != current_config.get("admin_ai_model"):
                update_model_for_role("admin_ai", admin_ai_model_id) 
                st.success(f"✅ Admin AI: {current_config.get('admin_ai_model')} → {admin_ai_model_id}")
                changes_made = True
            
            if changes_made:
                st.success("🔄 Configuration updated!")
                time.sleep(1)
                st.rerun()
            else:
                st.info("ℹ️ No changes to apply")
    
    with col2:
        if st.button("🔧 Test Models"):
            with st.spinner("Testing models..."):
                # Test search agent
                try:
                    test_messages = [{"role": "user", "content": "Say 'Hello from search agent'"}]
                    response = call_search_agent_model(test_messages)
                    st.success(f"✅ Search Agent: {response[:50]}...")
                except Exception as e:
                    st.error(f"❌ Search Agent failed: {str(e)}")
                
                # Test narrator
                try:
                    test_messages = [{"role": "user", "content": "Say 'Hello from narrator'"}]
                    response = call_narrator_model(test_messages, stream=False)
                    st.success(f"✅ Narrator: {response[:50]}...")
                except Exception as e:
                    st.error(f"❌ Narrator failed: {str(e)}")

                # Test admin_ai
                try:
                    test_messages = [{"role": "user", "content": "Say 'Hello from admin AI'"}]
                    response = call_admin_ai_model(test_messages)
                    st.success(f"✅ Admin AI: {response[:50]}...")
                except Exception as e:
                    st.error(f"❌ Admin AI failed: {str(e)}")
    
    with col3:
        if st.button("↩️ Reset to Defaults"):
            update_model_for_role("search_agent", "gpt-4o-mini")
            update_model_for_role("narrator", "gpt-4o-mini")
            update_model_for_role("admin_ai", "gpt-4o-mini")
            st.success("✅ Reset to defaults!")
            st.rerun()

# Export key functions
__all__ = [
    'load_model_config',
    'save_model_config', 
    'get_model_for_role',
    'update_model_for_role',
    'call_narrator_model',
    'call_search_agent_model',
    'call_admin_ai_model',
    'render_llm_model_config'
]
