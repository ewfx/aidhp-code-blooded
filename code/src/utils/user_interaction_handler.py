import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

class UserInteractionHandler:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize data files
        self._initialize_data_files()
        
        # Cache for user profiles and interactions
        self._profiles_cache = None
        self._interactions_cache = None
        self._last_cache_update = None
        self._cache_timeout = 300  # 5 minutes cache timeout
    
    def _initialize_data_files(self):
        """Initialize data files with proper schema"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Initialize profiles file with schema
            if not os.path.exists(self.profiles_file):
                profiles_df = pd.DataFrame(columns=[
                    'user_id', 'username', 'password_hash', 'email', 'created_at',
                    'income_level', 'risk_tolerance', 'investment_goals',
                    'time_horizon', 'preferred_products', 'last_login'
                ])
                profiles_df.to_csv(self.profiles_file, index=False)
                self.logger.info(f"Created new profiles file: {self.profiles_file}")
            
            # Initialize interactions file with schema
            if not os.path.exists(self.interactions_file):
                interactions_df = pd.DataFrame(columns=[
                    'user_id', 'timestamp', 'interaction_type', 'product_name',
                    'feedback_score', 'clicked', 'selected', 'features', 'tags'
                ])
                interactions_df.to_csv(self.interactions_file, index=False)
                self.logger.info(f"Created new interactions file: {self.interactions_file}")
        except Exception as e:
            self.logger.error(f"Error initializing data files: {str(e)}")
            raise
    
    @property
    def profiles_file(self) -> str:
        return 'data/user_profiles.csv'
    
    @property
    def interactions_file(self) -> str:
        return 'data/user_interactions.csv'
    
    @property
    def products_file(self) -> str:
        return 'data/financial_products.csv'
    
    def _should_update_cache(self) -> bool:
        """Check if cache needs to be updated"""
        if self._last_cache_update is None:
            return True
        return (datetime.now() - self._last_cache_update).total_seconds() > self._cache_timeout
    
    @lru_cache(maxsize=100)
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile with caching"""
        try:
            profiles_df = self._get_profiles_df()
            user_profile = profiles_df[profiles_df['user_id'] == user_id].iloc[0].to_dict()
            
            # Convert string lists back to lists, handling different data types safely
            for field in ['financial_goals', 'preferred_services']:
                if field in user_profile:
                    if pd.isna(user_profile[field]):
                        user_profile[field] = []
                    elif isinstance(user_profile[field], str):
                        # Only split if it's a non-empty string
                        user_profile[field] = user_profile[field].split(',') if user_profile[field].strip() else []
                    else:
                        user_profile[field] = []
            
            # Ensure other fields have default values if they're NaN
            default_values = {
                'income_level': 'Medium',
                'risk_tolerance': 'Moderate',
                'banking_frequency': 'Monthly',
                'time_horizon': 'Medium',
                'investment_goals': [],
                'preferred_products': []
            }
            
            for field, default in default_values.items():
                if field in user_profile and pd.isna(user_profile[field]):
                    user_profile[field] = default
            
            return user_profile
        except IndexError:
            self.logger.warning(f"No profile found for user {user_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting user profile: {str(e)}")
            return None
    
    def _get_profiles_df(self) -> pd.DataFrame:
        """Get profiles DataFrame with caching"""
        if self._should_update_cache() or self._profiles_cache is None:
            self._profiles_cache = pd.read_csv(self.profiles_file)
            self._last_cache_update = datetime.now()
        return self._profiles_cache
    
    def _get_interactions_df(self) -> pd.DataFrame:
        """Get interactions DataFrame with caching"""
        if self._should_update_cache() or self._interactions_cache is None:
            self._interactions_cache = pd.read_csv(self.interactions_file)
            self._last_cache_update = datetime.now()
        return self._interactions_cache
    
    def record_interaction(self, user_id: str, interaction_type: str, product_name: str,
                          feedback_score: float = None, clicked: bool = False, selected: bool = False):
        """Record user interaction with enhanced feedback tracking"""
        try:
            # Load existing interactions
            interactions_df = pd.read_csv(self.interactions_file)
            
            # Create new interaction record
            new_interaction = {
                'user_id': user_id,
                'timestamp': pd.Timestamp.now(),
                'interaction_type': interaction_type,
                'product_name': product_name,
                'feedback_score': feedback_score,
                'clicked': clicked,
                'selected': selected
            }
            
            # Append new interaction
            interactions_df = pd.concat([interactions_df, pd.DataFrame([new_interaction])], ignore_index=True)
            
            # Save updated interactions
            interactions_df.to_csv(self.interactions_file, index=False)
            
            # Update cache
            self._interactions_cache = interactions_df
            
            self.logger.info(f"Recorded interaction for user {user_id} with product {product_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error recording interaction: {str(e)}")
            return False
    
    def get_user_interactions(self, user_id: str) -> pd.DataFrame:
        """Get user interactions with caching"""
        try:
            interactions_df = self._get_interactions_df()
            return interactions_df[interactions_df['user_id'] == user_id]
        except Exception as e:
            self.logger.error(f"Error getting user interactions: {str(e)}")
            return pd.DataFrame()
    
    def get_user_feedback(self, user_id: str) -> pd.DataFrame:
        """Get user feedback with optimized filtering"""
        try:
            interactions_df = self._get_interactions_df()
            return interactions_df[
                (interactions_df['user_id'] == user_id) &
                (interactions_df['interaction_type'] == 'feedback')
            ]
        except Exception as e:
            self.logger.error(f"Error getting user feedback: {str(e)}")
            return pd.DataFrame()
    
    def update_user_profile(self, user_id: str, profile_data: Dict) -> bool:
        """Update user profile with optimized file handling"""
        try:
            profiles_df = self._get_profiles_df()
            
            # Convert lists to strings for storage
            if 'financial_goals' in profile_data:
                profile_data['financial_goals'] = ','.join(profile_data['financial_goals'])
            if 'preferred_services' in profile_data:
                profile_data['preferred_services'] = ','.join(profile_data['preferred_services'])
            
            # Add timestamp
            profile_data['last_updated'] = datetime.now().isoformat()
            
            # Update or append profile
            if user_id in profiles_df['user_id'].values:
                # Update existing profile
                user_idx = profiles_df[profiles_df['user_id'] == user_id].index[0]
                for key, value in profile_data.items():
                    if key in profiles_df.columns:
                        profiles_df.at[user_idx, key] = value
            else:
                # Create new profile
                profile_data['user_id'] = user_id
                profiles_df = pd.concat([profiles_df, pd.DataFrame([profile_data])], ignore_index=True)
            
            # Save to file
            profiles_df.to_csv(self.profiles_file, index=False)
            
            # Update cache
            self._profiles_cache = profiles_df
            self._last_cache_update = datetime.now()
            
            self.logger.info(f"Updated profile for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating user profile: {str(e)}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences with caching"""
        try:
            user_profile = self.get_user_profile(user_id)
            if user_profile:
                return {
                    'income_level': user_profile['income_level'],
                    'risk_tolerance': user_profile['risk_tolerance'],
                    'investment_goals': user_profile['investment_goals'],
                    'time_horizon': user_profile['time_horizon'],
                    'preferred_products': user_profile['preferred_products']
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            return {}
    
    def get_interaction_stats(self, user_id: str) -> Dict:
        """Get user interaction statistics with optimized calculations"""
        try:
            interactions_df = self._get_interactions_df()
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            return {
                'total_interactions': len(user_interactions),
                'total_feedback': len(user_interactions[user_interactions['interaction_type'] == 'feedback']),
                'avg_feedback_score': user_interactions['feedback_score'].mean(),
                'click_rate': user_interactions['clicked'].mean(),
                'selection_rate': user_interactions['selected'].mean()
            }
        except Exception as e:
            self.logger.error(f"Error getting interaction stats: {str(e)}")
            return {}
    
    def get_training_data(self):
        """Get training data for model retraining"""
        interactions_df = pd.read_csv(self.interactions_file)
        profiles_df = pd.read_csv(self.profiles_file)
        
        # Merge interactions with profiles
        training_data = pd.merge(
            interactions_df,
            profiles_df,
            on='user_id',
            how='left'
        )
        
        # Convert feedback_details string back to dict
        training_data['feedback_details'] = training_data['feedback_details'].apply(
            lambda x: eval(x) if pd.notna(x) else None
        )
        
        return training_data.to_dict('records')
    
    def get_product_interactions(self, product_name: str) -> pd.DataFrame:
        """Get all interactions for a specific product."""
        interactions_df = pd.read_csv(self.interactions_file)
        return interactions_df[interactions_df['product_name'] == product_name]
    
    def get_product_metrics(self, product_name: str) -> Dict[str, float]:
        """Calculate product performance metrics."""
        interactions = self.get_product_interactions(product_name)
        if interactions.empty:
            return {}
        
        return {
            'total_interactions': len(interactions),
            'avg_duration': interactions['duration'].mean() if 'duration' in interactions.columns else 0,
            'click_rate': interactions['clicked'].mean() if 'clicked' in interactions.columns else 0,
            'selection_rate': interactions['selected'].mean() if 'selected' in interactions.columns else 0,
            'avg_feedback_score': interactions['feedback_score'].mean() if 'feedback_score' in interactions.columns else 0
        }
    
    def get_filtered_products(self, user_id: str, products_df: pd.DataFrame) -> pd.DataFrame:
        """Get products filtered based on user feedback"""
        try:
            # Load user interactions
            interactions_df = pd.read_csv(self.interactions_file)
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            # Calculate average feedback scores for each product
            product_feedback = user_interactions.groupby('product_name')['feedback_score'].agg(['mean', 'count'])
            
            # Filter out products with consistently low feedback (below 0.3)
            low_feedback_products = product_feedback[product_feedback['mean'] < 0.3].index.tolist()
            
            # Filter out products with no feedback or low interaction count
            no_feedback_products = product_feedback[product_feedback['count'] < 2].index.tolist()
            
            # Combine products to exclude
            products_to_exclude = set(low_feedback_products + no_feedback_products)
            
            # Filter products
            filtered_products = products_df[~products_df['product_name'].isin(products_to_exclude)]
            
            return filtered_products
        except Exception as e:
            self.logger.error(f"Error filtering products: {str(e)}")
            return products_df

    def _safe_split(self, x):
        """Safely split a string into a list, handling non-string inputs"""
        if pd.isna(x) or not isinstance(x, str):
            return []
        return [item.strip() for item in x.split(',') if item.strip()]
    
    def _safe_len(self, x):
        """Safely get length of a list, handling non-list inputs"""
        return len(x) if isinstance(x, list) else 0
    
    def get_enhanced_training_data(self) -> pd.DataFrame:
        """Get enhanced training data with user profiles and product information"""
        try:
            # Load interaction data
            self.logger.info("Loading interaction data...")
            interactions_df = pd.read_csv(self.interactions_file)
            self.logger.info(f"Loaded {len(interactions_df)} interactions")
            
            # Load user profiles
            self.logger.info("Loading user profiles...")
            profiles_df = pd.read_csv(self.profiles_file)
            self.logger.info(f"Loaded {len(profiles_df)} profiles")
            
            # Load product data
            self.logger.info("Loading product data...")
            products_df = pd.read_csv(self.products_file)
            products_df = products_df.rename(columns={
                'features': 'product_features',
                'tags': 'product_tags'
            })
            self.logger.info(f"Loaded {len(products_df)} products")
            
            # Convert string lists to actual lists
            interactions_df['features'] = interactions_df['features'].apply(self._safe_split)
            interactions_df['tags'] = interactions_df['tags'].apply(self._safe_split)
            products_df['product_features'] = products_df['product_features'].apply(self._safe_split)
            products_df['product_tags'] = products_df['product_tags'].apply(self._safe_split)
            
            # Convert timestamps to datetime
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], format='mixed')
            
            # Merge data
            merged_df = interactions_df.merge(profiles_df, on='user_id', how='left')
            merged_df = merged_df.merge(products_df, on='product_name', how='left')
            
            # Add derived features
            merged_df['interaction_count'] = merged_df.groupby('user_id')['user_id'].transform('count')
            merged_df['avg_feedback'] = merged_df.groupby('user_id')['feedback_score'].transform('mean')
            merged_df['hour'] = merged_df['timestamp'].dt.hour
            merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
            
            # Count features and tags safely
            merged_df['num_interaction_features'] = merged_df['features'].apply(self._safe_len)
            merged_df['num_interaction_tags'] = merged_df['tags'].apply(self._safe_len)
            merged_df['num_product_features'] = merged_df['product_features'].apply(self._safe_len)
            merged_df['num_product_tags'] = merged_df['product_tags'].apply(self._safe_len)
            
            # Fill NaN values
            merged_df = merged_df.fillna({
                'income_level': 'Medium',
                'risk_tolerance': 'Moderate',
                'financial_goals': 'General',
                'preferred_services': 'Basic',
                'banking_frequency': 'Regular',
                'min_balance': 0,
                'interest_rate': 0,
                'annual_fee': 0,
                'monthly_fee': 0,
                'min_investment': 0,
                'management_fee': 0,
                'tax_benefits': 'None'
            })
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced training data: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def collect_recommendation_feedback(self, user_id: str, product_name: str, feedback_data: Dict) -> bool:
        """Collect detailed feedback for a recommended product"""
        try:
            # Validate feedback data
            required_fields = ['relevance', 'clarity', 'value', 'complexity', 'risk_level', 'cost']
            if not all(field in feedback_data for field in required_fields):
                self.logger.error("Missing required feedback fields")
                return False
            
            # Load existing feedback
            feedback_file = 'data/user_feedback.csv'
            if os.path.exists(feedback_file):
                feedback_df = pd.read_csv(feedback_file)
            else:
                feedback_df = pd.DataFrame(columns=[
                    'user_id', 'product_name', 'timestamp', 'relevance', 'clarity',
                    'value', 'complexity', 'risk_level', 'cost', 'overall_score'
                ])
            
            # Calculate overall score
            overall_score = sum(feedback_data[field] for field in required_fields) / len(required_fields)
            
            # Add new feedback
            new_feedback = pd.DataFrame({
                'user_id': [user_id],
                'product_name': [product_name],
                'timestamp': [datetime.now().isoformat()],
                'relevance': [feedback_data['relevance']],
                'clarity': [feedback_data['clarity']],
                'value': [feedback_data['value']],
                'complexity': [feedback_data['complexity']],
                'risk_level': [feedback_data['risk_level']],
                'cost': [feedback_data['cost']],
                'overall_score': [overall_score]
            })
            
            feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
            feedback_df.to_csv(feedback_file, index=False)
            
            # Update product feedback in interactions
            self.update_product_feedback(product_name, overall_score)
            
            self.logger.info(f"Feedback collected for user {user_id} and product {product_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {str(e)}")
            return False
    
    def get_user_feedback(self, user_id: str) -> Optional[pd.DataFrame]:
        """Get feedback history for a user"""
        try:
            feedback_file = 'data/user_feedback.csv'
            if not os.path.exists(feedback_file):
                return None
            
            feedback_df = pd.read_csv(feedback_file)
            user_feedback = feedback_df[feedback_df['user_id'] == user_id]
            
            return user_feedback if not user_feedback.empty else None
            
        except Exception as e:
            self.logger.error(f"Error getting user feedback: {str(e)}")
            return None
    
    def get_product_feedback(self, product_name: str) -> Optional[pd.DataFrame]:
        """Get feedback history for a product"""
        try:
            feedback_file = 'data/user_feedback.csv'
            if not os.path.exists(feedback_file):
                return None
            
            feedback_df = pd.read_csv(feedback_file)
            product_feedback = feedback_df[feedback_df['product_name'] == product_name]
            
            return product_feedback if not product_feedback.empty else None
            
        except Exception as e:
            self.logger.error(f"Error getting product feedback: {str(e)}")
            return None
    
    def update_product_feedback(self, product_name: str, feedback_score: float) -> bool:
        """Update product feedback score in interactions"""
        try:
            interactions_file = 'data/user_interactions.csv'
            if not os.path.exists(interactions_file):
                return False
            
            interactions_df = pd.read_csv(interactions_file)
            product_mask = interactions_df['product_name'] == product_name
            
            if product_mask.any():
                # Update feedback score for the product
                interactions_df.loc[product_mask, 'feedback_score'] = feedback_score
                interactions_df.to_csv(interactions_file, index=False)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating product feedback: {str(e)}")
            return False 