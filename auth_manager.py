"""
Authentication and User Session Management for RLS
Handles user login, session management, and user context for Row-Level Security
"""

import streamlit as st
import jwt
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from schema_registry import get_user_collections, get_rls_config

@dataclass
class UserContext:
    """User context for RLS and authentication."""
    user_id: str
    username: str
    role: str
    permissions: list
    accessible_collections: list
    session_id: str
    login_time: datetime
    expires_at: datetime
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['login_time'] = self.login_time.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserContext':
        """Create from dictionary."""
        data['login_time'] = datetime.fromisoformat(data['login_time'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at
    
    def should_bypass_rls(self) -> bool:
        """Check if user should bypass RLS based on role."""
        rls_config = get_rls_config()
        bypass_roles = rls_config.get("bypass_roles", ["admin", "super_user"])
        return self.role in bypass_roles

class AuthManager:
    """Manages authentication and user sessions."""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
        self.session_duration_hours = int(os.getenv("SESSION_DURATION_HOURS", "8"))
        
        # Simple user database - in production, use a real database
        self.users_db = {
            "admin": {
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "permissions": ["read", "write", "admin"],
                "user_id": "admin_001"
            },
            "user1": {
                "password_hash": self._hash_password("user123"),
                "role": "user",
                "permissions": ["read"],
                "user_id": "user_001"
            },
            "manager": {
                "password_hash": self._hash_password("manager123"),
                "role": "manager",
                "permissions": ["read", "write"],
                "user_id": "manager_001"
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> Optional[UserContext]:
        """Authenticate user and create session."""
        if username not in self.users_db:
            return None
        
        user_data = self.users_db[username]
        password_hash = self._hash_password(password)
        
        if password_hash != user_data["password_hash"]:
            return None
        
        # Get user's accessible collections
        accessible_collections = get_user_collections(user_data["user_id"])
        
        # Create user context
        now = datetime.now()
        user_id_val = user_data["user_id"]
        try:
            user_id_val = int(user_id_val)
        except Exception:
            pass

        user_context = UserContext(
            user_id=user_id_val,
            username=username,
            role=user_data["role"],
            permissions=user_data["permissions"],
            accessible_collections=accessible_collections,
            session_id=self._generate_session_id(username),
            login_time=now,
            expires_at=now + timedelta(hours=self.session_duration_hours)
        )
        
        return user_context
    
    def _generate_session_id(self, username: str) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        data = f"{username}:{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def create_jwt_token(self, user_context: UserContext) -> str:
        """Create JWT token for the user."""
        payload = {
            "user_id": user_context.user_id,
            "username": user_context.username,
            "role": user_context.role,
            "session_id": user_context.session_id,
            "exp": user_context.expires_at.timestamp()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# Streamlit session management
class StreamlitSessionManager:
    """Manages user sessions in Streamlit."""
    
    def __init__(self):
        self.auth_manager = AuthManager()
    
    def login_ui(self) -> Optional[UserContext]:
        """Display login UI and handle authentication."""
        st.title("🔐 Login Required")
        st.write("Please log in to access the application with Row-Level Security.")
        
        with st.form("login_form"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Demo Credentials:")
                st.code("""
Username: admin
Password: admin123
Role: Admin (bypasses RLS)

Username: user1  
Password: user123
Role: User (RLS applied)

Username: manager
Password: manager123
Role: Manager (RLS applied)
                """)
            
            with col2:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("🔑 Login", type="primary")
                
                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password")
                        return None
                    
                    user_context = self.auth_manager.authenticate(username, password)
                    if user_context:
                        # Store in session
                        st.session_state.user_context = user_context
                        st.session_state.authenticated = True
                        st.session_state.jwt_token = self.auth_manager.create_jwt_token(user_context)
                        
                        st.success(f"✅ Welcome {user_context.username}! Role: {user_context.role}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        return None
    
    def get_current_user(self) -> Optional[UserContext]:
        """Get current authenticated user from session."""
        if not st.session_state.get("authenticated", False):
            return None
        
        user_context = st.session_state.get("user_context")
        if not user_context:
            return None
        
        # Check if session is expired
        if user_context.is_expired():
            self.logout()
            return None
        
        return user_context
    
    def logout(self):
        """Logout current user."""
        st.session_state.authenticated = False
        st.session_state.user_context = None
        st.session_state.jwt_token = None
        st.rerun()
    
    def require_authentication(self) -> UserContext:
        """Require authentication before proceeding."""
        user_context = self.get_current_user()
        
        if not user_context:
            self.login_ui()
            st.stop()
        
        return user_context
    
    def show_user_info_sidebar(self, user_context: UserContext):
        """Show user info in sidebar."""
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Current User")
            st.write(f"**User:** {user_context.username}")
            st.write(f"**Role:** {user_context.role}")
            st.write(f"**User ID:** {user_context.user_id}")
            
            # Show RLS status
            if user_context.should_bypass_rls():
                st.success("🔓 RLS Bypass (Admin)")
            else:
                st.info("🔒 RLS Active")
            
            # Show accessible collections
            if user_context.accessible_collections:
                st.write(f"**Collections:** {len(user_context.accessible_collections)}")
                with st.expander("View Collections"):
                    for coll in user_context.accessible_collections:
                        st.caption(f"📄 {coll}")
            
            # Session info
            remaining_time = user_context.expires_at - datetime.now()
            hours_remaining = remaining_time.total_seconds() / 3600
            st.caption(f"Session expires in {hours_remaining:.1f} hours")
            
            if st.button("🚪 Logout"):
                self.logout()

# Global session manager instance
session_manager = StreamlitSessionManager()

# Helper functions for easy integration
def get_current_user_id() -> Optional[str]:
    """Get current user ID for RLS filtering."""
    user_context = session_manager.get_current_user()
    return user_context.user_id if user_context else None

def get_current_user_context() -> Optional[UserContext]:
    """Get full user context."""
    return session_manager.get_current_user()

def require_auth() -> UserContext:
    """Require authentication and return user context."""
    return session_manager.require_authentication()
