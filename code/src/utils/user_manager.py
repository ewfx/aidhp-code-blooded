import pandas as pd
import os
from datetime import datetime
import hashlib
import uuid
import logging

class UserManager:
    def __init__(self):
        # Set up logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Then initialize the users file
        self.users_file = 'data/users.csv'
        self._initialize_users_file()
    
    def _initialize_users_file(self):
        """Initialize users file if it doesn't exist"""
        try:
            os.makedirs('data', exist_ok=True)
            
            if not os.path.exists(self.users_file):
                pd.DataFrame(columns=[
                    'user_id', 'username', 'email', 'password_hash',
                    'created_at', 'last_login'
                ]).to_csv(self.users_file, index=False)
                self.logger.info(f"Created new users file at {self.users_file}")
            else:
                self.logger.info(f"Using existing users file at {self.users_file}")
        except Exception as e:
            self.logger.error(f"Error initializing users file: {str(e)}")
            raise
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password):
        """Create a new user"""
        try:
            users_df = pd.read_csv(self.users_file)
            self.logger.info(f"Current users in database: {len(users_df)}")
            
            # Check if username or email already exists
            if username in users_df['username'].values:
                self.logger.warning(f"Username '{username}' already exists")
                return False, "Username already exists"
            if email in users_df['email'].values:
                self.logger.warning(f"Email '{email}' already exists")
                return False, "Email already exists"
            
            # Create new user
            new_user = {
                'user_id': str(uuid.uuid4()),
                'username': username,
                'email': email,
                'password_hash': self._hash_password(password),
                'created_at': datetime.now().isoformat(),
                'last_login': None
            }
            
            users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
            users_df.to_csv(self.users_file, index=False)
            self.logger.info(f"Created new user: {username}")
            
            return True, new_user['user_id']
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            return False, f"Error creating user: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate a user"""
        try:
            users_df = pd.read_csv(self.users_file)
            self.logger.info(f"Attempting to authenticate user: {username}")
            
            # Find user
            user = users_df[users_df['username'] == username]
            if user.empty:
                self.logger.warning(f"User not found: {username}")
                return False, "User not found"
            
            # Check password
            if user['password_hash'].iloc[0] != self._hash_password(password):
                self.logger.warning(f"Invalid password for user: {username}")
                return False, "Invalid password"
            
            # Update last login
            users_df.loc[users_df['username'] == username, 'last_login'] = datetime.now().isoformat()
            users_df.to_csv(self.users_file, index=False)
            self.logger.info(f"Successfully authenticated user: {username}")
            
            return True, user['user_id'].iloc[0]
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            return False, f"Error authenticating user: {str(e)}"
    
    def get_user_info(self, user_id):
        """Get user information"""
        try:
            users_df = pd.read_csv(self.users_file)
            user = users_df[users_df['user_id'] == user_id]
            
            if user.empty:
                self.logger.warning(f"User info not found for ID: {user_id}")
                return None
            
            return user.iloc[0].to_dict()
        except Exception as e:
            self.logger.error(f"Error getting user info: {str(e)}")
            return None
    
    def update_user_info(self, user_id, **kwargs):
        """Update user information"""
        try:
            users_df = pd.read_csv(self.users_file)
            
            if user_id not in users_df['user_id'].values:
                self.logger.warning(f"User not found for update: {user_id}")
                return False, "User not found"
            
            # Update fields
            for key, value in kwargs.items():
                if key in users_df.columns and key not in ['user_id', 'password_hash']:
                    users_df.loc[users_df['user_id'] == user_id, key] = value
            
            users_df.to_csv(self.users_file, index=False)
            self.logger.info(f"Updated user info for ID: {user_id}")
            return True, "User information updated successfully"
        except Exception as e:
            self.logger.error(f"Error updating user info: {str(e)}")
            return False, f"Error updating user info: {str(e)}" 