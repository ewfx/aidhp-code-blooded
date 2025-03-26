import os
from data_processor import DataProcessor
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def train_product_classifier():
    """Train a classifier to categorize financial products"""
    processor = DataProcessor()
    
    # Load and preprocess data
    df = processor.load_financial_products('../data/financial_products.csv')
    
    # Create product embeddings
    product_embeddings = processor.create_product_embeddings(df)
    
    # Create labels (using tags as categories)
    all_tags = set()
    for tags in df['tags']:
        all_tags.update(tags)
    
    # Create binary labels for each tag
    for tag in all_tags:
        df[f'is_{tag}'] = df['tags'].apply(lambda x: 1 if tag in x else 0)
    
    # Prepare features and labels
    X = product_embeddings
    y_columns = [col for col in df.columns if col.startswith('is_')]
    y = df[y_columns]
    
    # Train a classifier for each tag
    classifiers = {}
    for col in y_columns:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y[col])
        classifiers[col] = clf
        processor.save_model(clf, f'product_classifier_{col}')
    
    return classifiers

def train_sentiment_analyzer():
    """Fine-tune the sentiment analyzer on financial domain data"""
    # This would require a financial domain sentiment dataset
    # For now, we'll use the pre-trained model
    pass

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Train models
    print("Training product classifiers...")
    classifiers = train_product_classifier()
    print("Training completed!")
    
    print("Models saved in the 'models' directory.")

if __name__ == "__main__":
    main() 