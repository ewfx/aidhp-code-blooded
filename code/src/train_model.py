import pandas as pd
from utils.model_utils import HuggingFaceModels
from utils.user_interaction_handler import UserInteractionHandler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """Train the recommendation model with new data"""
    try:
        # Initialize models and handlers
        hf_models = HuggingFaceModels()
        interaction_handler = UserInteractionHandler()
        
        # Get enhanced training data
        logger.info("Loading training data...")
        training_data = interaction_handler.get_enhanced_training_data()
        
        if training_data.empty:
            logger.warning("No training data available. Please add some user interactions first.")
            return
        
        # Train the model
        logger.info("Training recommendation model...")
        mse, r2 = hf_models.retrain_model(training_data)
        
        logger.info(f"Model training completed!")
        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

if __name__ == "__main__":
    train_model() 