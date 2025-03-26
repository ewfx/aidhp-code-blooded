import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.model_utils import HuggingFaceModels
from utils.user_interaction_handler import UserInteractionHandler
from utils.user_manager import UserManager
import os
from dotenv import load_dotenv
import uuid
from typing import Dict, List
import asyncio

# Load environment variables
load_dotenv()

# Initialize session state for model
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.hf_models = None

def load_models():
    """Load all necessary models and handlers"""
    if not st.session_state.model_initialized:
        st.session_state.hf_models = HuggingFaceModels()
        st.session_state.model_initialized = True
    
    data_processor = DataProcessor()
    interaction_handler = UserInteractionHandler()
    user_manager = UserManager()
    return st.session_state.hf_models, data_processor, interaction_handler, user_manager

def login_page():
    """Display login page"""
    st.title("🏦 Banking Personalization System")
    
    # Load user manager
    _, _, _, user_manager = load_models()
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                success, result = user_manager.authenticate_user(username, password)
                if success:
                    st.session_state.user_id = result
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(result)
    
    with tab2:
        st.subheader("Create New Account")
        new_username = st.text_input("Choose Username")
        email = st.text_input("Email")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Sign Up"):
            if not new_username or not email or not new_password or not confirm_password:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                success, result = user_manager.create_user(new_username, email, new_password)
                if success:
                    st.success("Account created successfully! Please login.")
                    st.rerun()
                else:
                    st.error(result)

async def get_personalized_recommendations(user_id: str, n_recommendations: int = 5) -> List[Dict]:
    """Get personalized product recommendations"""
    try:
        # Load models
        hf_models, _, _, _ = load_models()
        
        # Get recommendations using the async method
        recommendations = await hf_models.get_recommendations(user_id, n_recommendations)
        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

def main():
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # Show login page if user is not authenticated
    if not st.session_state.user_id:
        login_page()
        return
    
    # Load models and handlers
    hf_models, data_processor, interaction_handler, user_manager = load_models()
    
    # Add logout button in sidebar
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
    
    st.title("🏦 Banking Personalization System")
    
    # Load financial products
    df = data_processor.load_financial_products('data/financial_products.csv')
    
    # Filter products based on user feedback
    filtered_df = interaction_handler.get_filtered_products(st.session_state.user_id, df)
    
    # User Preferences
    st.header("Your Financial Preferences")
    
    # Load existing user preferences
    existing_profile = interaction_handler.get_user_profile(st.session_state.user_id)
    if existing_profile is None:
        existing_profile = {
            'income_level': 'Medium',
            'risk_tolerance': 'Moderate',
            'financial_goals': [],
            'preferred_services': [],
            'banking_frequency': 'Monthly'
        }
    
    # Income Level
    income_level = st.selectbox(
        "What is your income level?",
        ["Low", "Medium", "High", "Very High"],
        index=["Low", "Medium", "High", "Very High"].index(existing_profile.get('income_level', 'Medium'))
    )
    
    # Risk Tolerance
    risk_tolerance = st.selectbox(
        "What is your risk tolerance?",
        ["Conservative", "Moderate", "Aggressive", "Very Aggressive"],
        index=["Conservative", "Moderate", "Aggressive", "Very Aggressive"].index(existing_profile.get('risk_tolerance', 'Moderate'))
    )
    
    # Financial Goals
    financial_goals_options = ["Save for Retirement", "Build Emergency Fund", "Invest for Growth", 
                              "Save for Home", "Pay Off Debt", "Generate Passive Income"]
    existing_goals = existing_profile.get('financial_goals', [])
    valid_goals = [goal for goal in existing_goals if goal in financial_goals_options]
    financial_goals = st.multiselect(
        "What are your financial goals?",
        financial_goals_options,
        default=valid_goals
    )
    
    # Preferred Services
    preferred_services_options = [
        # Banking Services
        "Checking Account", "Savings Account", "Money Market Account",
        "Certificate of Deposit (CD)", "Online Banking", "Mobile Banking",
        
        # Investment Services
        "Stocks", "Mutual Funds", "Exchange-Traded Funds (ETFs)",
        "Bonds", "Retirement Accounts", "Robo-Advisory Services",
        
        # Credit Services
        "Credit Cards", "Personal Loans", "Home Loans",
        "Auto Loans", "Student Loans", "Business Loans",
        
        # Insurance Services
        "Life Insurance", "Health Insurance", "Auto Insurance",
        "Home Insurance", "Disability Insurance", "Long-term Care Insurance",
        
        # Wealth Management
        "Financial Planning", "Estate Planning", "Tax Planning",
        "Investment Advisory", "Portfolio Management", "Trust Services"
    ]
    existing_services = existing_profile.get('preferred_services', [])
    valid_services = [service for service in existing_services if service in preferred_services_options]
    preferred_services = st.multiselect(
        "Which financial services are you interested in?",
        preferred_services_options,
        default=valid_services
    )
    
    # Banking Frequency
    banking_frequency = st.selectbox(
        "How often do you use banking services?",
        ["Rarely", "Monthly", "Weekly", "Daily"],
        index=["Rarely", "Monthly", "Weekly", "Daily"].index(existing_profile.get('banking_frequency', 'Monthly'))
    )
    
    # Create user profile
    user_preferences = {
        "income_level": income_level,
        "risk_tolerance": risk_tolerance,
        "financial_goals": financial_goals,
        "preferred_services": preferred_services,
        "banking_frequency": banking_frequency
    }
    
    # Update user profile in database
    if interaction_handler.update_user_profile(st.session_state.user_id, user_preferences):
        st.success("Your preferences have been updated!")
    
    # Add a button to refresh recommendations
    if st.button("Refresh Recommendations"):
        st.rerun()
    
    # Get personalized recommendations
    st.header("Personalized Recommendations")
    
    # Show loading spinner while getting recommendations
    with st.spinner("Getting personalized recommendations..."):
        recommendations = asyncio.run(get_personalized_recommendations(st.session_state.user_id))
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['product_name']} (Score: {rec['predicted_score']:.2f})"):
                st.write(rec['description'])
    else:
        st.warning("No recommendations available yet. Please interact with some products to get personalized recommendations.")

if __name__ == "__main__":
    main() 