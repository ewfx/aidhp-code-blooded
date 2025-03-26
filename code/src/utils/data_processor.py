import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import joblib
import os

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.product_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
    def load_financial_products(self, file_path):
        """Load and preprocess financial products data"""
        df = pd.read_csv(file_path)
        # Convert string lists to actual lists
        df['features'] = df['features'].apply(lambda x: x.split(','))
        df['tags'] = df['tags'].apply(lambda x: x.split(','))
        return df
    
    def create_product_embeddings(self, df):
        """Create embeddings for product descriptions"""
        product_texts = []
        for _, row in df.iterrows():
            text = f"{row['product_name']} {row['description']} {' '.join(row['features'])} {' '.join(row['tags'])}"
            product_texts.append(text)
        
        embeddings = self.sentence_model.encode(product_texts)
        return embeddings
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of user feedback or product descriptions"""
        return self.sentiment_analyzer(text)
    
    def classify_product(self, product_description, candidate_labels):
        """Classify product into predefined categories"""
        return self.product_classifier(product_description, candidate_labels)
    
    def scale_numerical_features(self, df):
        """Scale numerical features for better comparison"""
        numerical_columns = ['min_balance', 'interest_rate', 'annual_fee', 'monthly_fee', 'min_investment', 'management_fee']
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df
    
    def save_model(self, model, model_name):
        """Save trained model"""
        model_path = os.path.join('models', f'{model_name}.joblib')
        joblib.dump(model, model_path)
    
    def load_model(self, model_name):
        """Load trained model"""
        model_path = os.path.join('models', f'{model_name}.joblib')
        return joblib.load(model_path)
    
    def create_user_profile_embedding(self, user_preferences):
        """Create embedding for user preferences"""
        return self.sentence_model.encode([user_preferences])[0]
    
    def calculate_product_similarity(self, user_embedding, product_embeddings):
        """Calculate cosine similarity between user profile and products"""
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([user_embedding], product_embeddings)[0]
        return similarities 