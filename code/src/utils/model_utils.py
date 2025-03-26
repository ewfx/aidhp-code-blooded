from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from utils.user_interaction_handler import UserInteractionHandler
import asyncio

# Load environment variables
load_dotenv()

class HuggingFaceModels:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.models_dir = 'models'
        self.metrics_path = os.path.join(self.models_dir, 'model_metrics.csv')
        self.last_training_path = os.path.join(self.models_dir, 'last_training.txt')
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize model paths
        self.recommendation_model_path = os.path.join(self.models_dir, 'recommendation_model.joblib')
        
        # Initialize metrics file if it doesn't exist
        self._initialize_metrics()
        
        # Get Hugging Face token
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            self.logger.warning("Hugging Face token not found in environment variables. Some features may be limited.")
        
        # Initialize models
        self._initialize_models()
        
        # Financial domain specific labels
        self.financial_labels = [
            "savings", "investment", "credit", "insurance", "retirement",
            "business", "student", "premium", "basic", "digital"
        ]
        
        # Financial sentiment labels
        self.sentiment_labels = [
            "positive", "negative", "neutral", "high-risk", "low-risk",
            "high-reward", "low-reward", "complex", "simple"
        ]
        
        # Detailed feedback categories
        self.feedback_categories = {
            "relevance": "How relevant is this recommendation?",
            "clarity": "How clear is the product description?",
            "value": "How valuable is this product for your needs?",
            "complexity": "How appropriate is the complexity level?",
            "risk_level": "How appropriate is the risk level?",
            "cost": "How appropriate is the cost structure?"
        }
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.recommendation_model_path
        self.last_training_time = None
        self.training_interval = 3600  # 1 hour in seconds
        self.is_training = False
        self._initialize_model()
    
    def _initialize_metrics(self):
        """Initialize metrics file if it doesn't exist"""
        if not os.path.exists(self.metrics_path):
            metrics_df = pd.DataFrame(columns=['timestamp', 'mse', 'r2'])
            metrics_df.to_csv(self.metrics_path, index=False)
            self.logger.info(f"Created new metrics file: {self.metrics_path}")
    
    def _initialize_models(self):
        """Initialize all models at startup"""
        try:
            self.logger.info("Initializing models...")
            
            # Initialize embedding model
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model initialized")
            
            # Initialize sentiment analyzer with token if available
            if self.hf_token:
                self._sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    token=self.hf_token
                )
                self.logger.info("Sentiment analyzer initialized")
            else:
                self._sentiment_analyzer = None
                self.logger.warning("Sentiment analyzer not initialized due to missing token")
            
            # Initialize classifier with token if available
            if self.hf_token:
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    token=self.hf_token
                )
                self.logger.info("Classifier initialized")
            else:
                self._classifier = None
                self.logger.warning("Classifier not initialized due to missing token")
            
            # Initialize recommendation model
            if os.path.exists(self.recommendation_model_path):
                self._recommendation_model = joblib.load(self.recommendation_model_path)
                self.logger.info("Recommendation model loaded")
            else:
                self._recommendation_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.logger.info("New recommendation model initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    @property
    def embedding_model(self):
        """Get embedding model"""
        return self._embedding_model
    
    @property
    def sentiment_analyzer(self):
        """Get sentiment analyzer"""
        return self._sentiment_analyzer
    
    @property
    def classifier(self):
        """Get classifier"""
        return self._classifier
    
    @property
    def recommendation_model(self):
        """Get recommendation model"""
        return self._recommendation_model
    
    @lru_cache(maxsize=1000)
    def create_user_profile_embedding(self, profile_text: str) -> np.ndarray:
        """Create embedding for user preferences with caching"""
        return self.embedding_model.encode(profile_text)
    
    def create_user_profile_embedding_from_dict(self, user_preferences: Dict) -> np.ndarray:
        """Create embedding for user preferences from dictionary"""
        profile_text = f"income: {user_preferences.get('income_level', '')}, "
        profile_text += f"risk: {user_preferences.get('risk_tolerance', '')}, "
        profile_text += f"goals: {user_preferences.get('investment_goals', '')}, "
        profile_text += f"horizon: {user_preferences.get('time_horizon', '')}"
        return self.create_user_profile_embedding(profile_text)
    
    @lru_cache(maxsize=1000)
    def create_product_embedding(self, product_description: str) -> np.ndarray:
        """Create embedding for a product with caching"""
        return self.embedding_model.encode(product_description)
    
    def create_product_embedding_from_dict(self, product_dict: Dict) -> np.ndarray:
        """Create embedding for product from dictionary"""
        product_text = f"{product_dict.get('name', '')} {' '.join(product_dict.get('features', []))} {' '.join(product_dict.get('tags', []))}"
        return self.create_product_embedding(product_text)
    
    @lru_cache(maxsize=1000)
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment with caching"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    @lru_cache(maxsize=1000)
    def classify_product(self, product_description: str, candidate_labels: List[str]) -> Dict:
        """Classify product with caching"""
        try:
            result = self.classifier(product_description, candidate_labels)
            return {
                'labels': result['labels'],
                'scores': result['scores']
            }
        except Exception as e:
            self.logger.error(f"Error classifying product: {str(e)}")
            return {'labels': [], 'scores': []}
    
    def analyze_complexity(self, product_description: str) -> Dict:
        """Analyze product complexity"""
        try:
            # Use sentence length and word complexity as proxies for complexity
            sentences = product_description.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            
            # Simple word complexity check (can be enhanced)
            words = product_description.lower().split()
            complex_words = [w for w in words if len(w) > 8]
            complexity_score = len(complex_words) / len(words)
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'complexity_score': complexity_score,
                'complexity_level': 'High' if complexity_score > 0.3 else 'Medium' if complexity_score > 0.15 else 'Low'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing complexity: {str(e)}")
            return {'avg_sentence_length': 0, 'complexity_score': 0, 'complexity_level': 'Unknown'}
    
    def analyze_risk_level(self, product_description: str) -> Dict:
        """Analyze product risk level"""
        try:
            # Use sentiment analysis and keyword matching for risk assessment
            sentiment = self.analyze_sentiment(product_description)
            
            # Risk-related keywords
            high_risk_keywords = ['high risk', 'volatile', 'speculative', 'uncertain']
            low_risk_keywords = ['stable', 'secure', 'guaranteed', 'conservative']
            
            text_lower = product_description.lower()
            high_risk_count = sum(1 for word in high_risk_keywords if word in text_lower)
            low_risk_count = sum(1 for word in low_risk_keywords if word in text_lower)
            
            # Calculate risk score
            risk_score = (high_risk_count - low_risk_count) / (len(high_risk_keywords) + len(low_risk_keywords))
            risk_score = max(min(risk_score, 1), 0)  # Normalize between 0 and 1
            
            return {
                'risk_score': risk_score,
                'risk_level': 'High' if risk_score > 0.6 else 'Medium' if risk_score > 0.3 else 'Low',
                'sentiment': sentiment
            }
        except Exception as e:
            self.logger.error(f"Error analyzing risk level: {str(e)}")
            return {'risk_score': 0.5, 'risk_level': 'Unknown', 'sentiment': {'label': 'NEUTRAL', 'score': 0.5}}
    
    def calculate_product_similarity(self, user_embedding: np.ndarray, product_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between user profile and product"""
        try:
            similarity = np.dot(user_embedding, product_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(product_embedding)
            )
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def get_personalized_recommendations(self, user_embedding: np.ndarray, 
                                      product_embeddings: List[np.ndarray],
                                      max_recommendations: int = 8) -> List[Tuple[int, float]]:
        """Get personalized product recommendations based on user embedding and product embeddings"""
        try:
            # Calculate cosine similarity between user and all products
            similarities = []
            for i, product_embedding in enumerate(product_embeddings):
                similarity = cosine_similarity(user_embedding.reshape(1, -1), 
                                            product_embedding.reshape(1, -1))[0][0]
                similarities.append((i, similarity))
            
            # Sort by similarity score in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            top_recommendations = similarities[:max_recommendations]
            
            return top_recommendations
        except Exception as e:
            self.logger.error(f"Error getting personalized recommendations: {str(e)}")
            return []
    
    def save_model(self, model, model_name: str) -> bool:
        """Save a trained model"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # Update last training timestamp
            with open(self.last_training_path, 'w') as f:
                f.write(datetime.now().isoformat())
            
            self.logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_name: str):
        """Load a trained model"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def _save_metrics(self, mse: float, r2: float):
        """Save model performance metrics"""
        try:
            metrics_df = pd.read_csv(self.metrics_path)
            new_metrics = pd.DataFrame({
                'timestamp': [datetime.now().isoformat()],
                'mse': [mse],
                'r2': [r2]
            })
            metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)
            metrics_df.to_csv(self.metrics_path, index=False)
            self.logger.info("Saved model metrics")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
    
    def should_retrain(self) -> bool:
        """Check if model needs retraining"""
        try:
            if not os.path.exists(self.last_training_path):
                return True
            
            with open(self.last_training_path, 'r') as f:
                last_training = datetime.fromisoformat(f.read().strip())
            
            # Retrain if more than 24 hours have passed
            return (datetime.now() - last_training).total_seconds() > 24 * 3600
        except Exception as e:
            self.logger.error(f"Error checking retrain status: {str(e)}")
            return True
    
    def get_detailed_feedback(self) -> Dict[str, str]:
        """Get detailed feedback categories and descriptions."""
        return self.feedback_categories
    
    def retrain_model(self, training_data):
        """Retrain the recommendation model with new data"""
        if training_data.empty:
            return
        
        # Convert training data to DataFrame
        training_df = pd.DataFrame(training_data)
        
        # Prepare features and labels
        X = []  # Features
        y = []  # Labels (feedback scores)
        
        for _, row in training_df.iterrows():
            try:
                # Create user profile embedding
                user_prefs = {
                    'income_level': row.get('income_level', 'Medium'),
                    'risk_tolerance': row.get('risk_tolerance', 'Moderate'),
                    'investment_goals': row.get('investment_goals', '').split(',') if isinstance(row.get('investment_goals'), str) else [],
                    'time_horizon': row.get('time_horizon', ''),
                    'preferred_services': row.get('preferred_services', ''),
                    'banking_frequency': row.get('banking_frequency', 'Regular')
                }
                user_embedding = self.create_user_profile_embedding_from_dict(user_prefs)
                
                # Create product embedding with default values
                product_dict = {
                    'name': row.get('product_name', ''),
                    'features': row.get('features', '').split(',') if isinstance(row.get('features'), str) else [],
                    'tags': row.get('tags', '').split(',') if isinstance(row.get('tags'), str) else []
                }
                product_embedding = self.create_product_embedding_from_dict(product_dict)
                
                # Combine embeddings
                combined_features = np.concatenate([user_embedding, product_embedding])
                X.append(combined_features)
                y.append(row.get('feedback_score', 0.5))  # Default to neutral feedback if missing
                
            except Exception as e:
                self.logger.error(f"Error processing row: {str(e)}")
                continue
        
        if not X or not y:
            self.logger.warning("No valid training data after processing")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.recommendation_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.recommendation_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        self.save_model(self.recommendation_model, 'recommendation_model')
        
        # Save metrics
        self._save_metrics(mse, r2)
        
        return mse, r2
    
    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            # Initialize model
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model.to(self.device)
            
            # Load existing model if available
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.logger.info("Loaded existing model from disk")
            
            # Load last training time
            if os.path.exists(self.metrics_path):
                metrics_df = pd.read_csv(self.metrics_path)
                if not metrics_df.empty:
                    self.last_training_time = pd.to_datetime(metrics_df['timestamp'].iloc[-1])
                    self.logger.info(f"Last training time: {self.last_training_time}")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
    
    async def train_model(self, force: bool = False) -> Dict:
        """Train the recommendation model"""
        try:
            # Check if training is already in progress
            if self.is_training:
                self.logger.info("Training already in progress")
                return {"status": "training_in_progress"}

            # Check if training is needed
            current_time = datetime.now()
            if not force and self.last_training_time:
                time_since_last_training = (current_time - self.last_training_time).total_seconds()
                if time_since_last_training < self.training_interval:
                    self.logger.info("Training not needed yet")
                    return {"status": "not_needed"}

            self.is_training = True
            self.logger.info("Starting model training...")

            # Get training data
            interaction_handler = UserInteractionHandler()
            training_data = interaction_handler.get_enhanced_training_data()
            
            if training_data is None or training_data.empty:
                self.logger.warning("No training data available")
                self.is_training = False
                return {"status": "no_data"}

            # Prepare features and target
            X = training_data.drop(['feedback_score', 'user_id', 'product_name', 'timestamp'], axis=1)
            y = training_data['feedback_score']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save model and metrics
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            metrics_df = pd.DataFrame({
                'timestamp': [current_time],
                'mse': [mse],
                'r2': [r2]
            })
            
            if os.path.exists(self.metrics_path):
                existing_metrics = pd.read_csv(self.metrics_path)
                metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)
            
            metrics_df.to_csv(self.metrics_path, index=False)
            self.last_training_time = current_time

            self.logger.info(f"Model training completed. MSE: {mse:.4f}, R²: {r2:.4f}")
            self.is_training = False
            return {
                "status": "success",
                "mse": mse,
                "r2": r2
            }

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            self.is_training = False
            return {"status": "error", "message": str(e)}
    
    async def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Get personalized product recommendations for a user"""
        try:
            # Get user profile and interaction data
            interaction_handler = UserInteractionHandler()
            user_profile = interaction_handler.get_user_profile(user_id)
            
            if user_profile is None:
                self.logger.warning(f"No profile found for user {user_id}")
                return []

            # Get user interactions and feedback
            interactions = interaction_handler.get_user_interactions(user_id)
            feedback = interaction_handler.get_user_feedback(user_id)
            
            # Create user profile text for embedding
            user_text = f"income: {user_profile.get('income_level', 'Medium')}, "
            user_text += f"risk: {user_profile.get('risk_tolerance', 'Moderate')}, "
            user_text += f"goals: {user_profile.get('investment_goals', '')}, "
            user_text += f"horizon: {user_profile.get('time_horizon', '')}, "
            user_text += f"services: {user_profile.get('preferred_services', '')}, "
            user_text += f"frequency: {user_profile.get('banking_frequency', 'Monthly')}"
            
            # Get user profile embedding
            user_embedding = self.create_user_profile_embedding(user_text)
            
            # Get all available products
            products_df = pd.read_csv('data/financial_products.csv')
            
            # Check if model needs training
            if not self.last_training_time or \
               (datetime.now() - self.last_training_time).total_seconds() >= self.training_interval:
                await self.train_model()
            
            if self.model is None:
                self.logger.warning("Model not initialized")
                return []
            
            # Generate predictions for all products
            predictions = []
            for _, product in products_df.iterrows():
                try:
                    # Create product text for embedding
                    product_text = f"{product['product_name']} {product['description']} "
                    product_text += f"features: {product['features']} tags: {product['tags']}"
                    
                    # Get product embedding
                    product_embedding = self.create_product_embedding(product_text)
                    
                    # Combine embeddings
                    combined_features = np.concatenate([user_embedding, product_embedding])
                    
                    # Make prediction
                    prediction = self.model.predict(combined_features.reshape(1, -1))[0]
                    
                    # Get product feedback if available
                    product_feedback = None
                    if feedback is not None and not feedback.empty:
                        product_feedback = feedback[feedback['product_name'] == product['product_name']]
                    
                    # Analyze product complexity and risk level
                    complexity = self.analyze_complexity(product['description'])
                    risk_level = self.analyze_risk_level(product['description'])
                    
                    # Create recommendation with feedback and analysis
                    recommendation = {
                        'product_name': product['product_name'],
                        'predicted_score': float(prediction),
                        'description': product['description'],
                        'complexity': complexity,
                        'risk_level': risk_level,
                        'feedback': {
                            'average_score': float(product_feedback['feedback_score'].mean()) if product_feedback is not None and not product_feedback.empty else None,
                            'total_feedback': len(product_feedback) if product_feedback is not None else 0,
                            'categories': self.get_detailed_feedback()
                        }
                    }
                    
                    predictions.append(recommendation)
                except Exception as e:
                    self.logger.error(f"Error making prediction for product {product['product_name']}: {str(e)}")
                    continue

            # Sort by predicted score and return top N
            recommendations = sorted(predictions, key=lambda x: x['predicted_score'], reverse=True)
            return recommendations[:n_recommendations]

        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def get_model_metrics(self) -> Dict:
        """Get the latest model metrics"""
        try:
            if os.path.exists(self.metrics_path):
                metrics_df = pd.read_csv(self.metrics_path)
                if not metrics_df.empty:
                    return {
                        'mse': metrics_df['mse'].iloc[-1],
                        'r2': metrics_df['r2'].iloc[-1],
                        'last_training': metrics_df['timestamp'].iloc[-1]
                    }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            return {} 