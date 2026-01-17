"""
User Authentication & Profile Management
"""
import json
import hashlib
import os
from pathlib import Path

class UserManager:
    def __init__(self):
        self.users_dir = Path.home() / ".orchestra" / "users"
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.auth_file = Path.home() / ".orchestra" / "auth.json"
        
        # Create auth file if it doesn't exist
        if not self.auth_file.exists():
            self.auth_file.write_text(json.dumps({}))
    
    def hash_password(self, password):
        """Simple hash for local-only use"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, password, profile_data):
        """Create new user account"""
        # Load existing users
        users = json.loads(self.auth_file.read_text())
        
        if username in users:
            return False, "Username already exists"
        
        # Create user directory structure
        user_path = self.users_dir / username
        user_path.mkdir(exist_ok=True)
        (user_path / "memory").mkdir(exist_ok=True)
        (user_path / "documents").mkdir(exist_ok=True)
        
        # Save password hash
        users[username] = self.hash_password(password)
        self.auth_file.write_text(json.dumps(users, indent=2))
        
        # Save profile
        profile_file = user_path / "profile.json"
        profile_file.write_text(json.dumps(profile_data, indent=2))
        
        return True, "User created successfully"
    
    def authenticate(self, username, password):
        """Verify username and password"""
        users = json.loads(self.auth_file.read_text())
        
        if username not in users:
            return False, "User not found"
        
        if users[username] != self.hash_password(password):
            return False, "Invalid password"
        
        return True, "Login successful"
    
    def get_profile(self, username):
        """Get user profile"""
        profile_file = self.users_dir / username / "profile.json"
        if profile_file.exists():
            return json.loads(profile_file.read_text())
        return None
