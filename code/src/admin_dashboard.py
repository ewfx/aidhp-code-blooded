import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.user_interaction_handler import UserInteractionHandler
from utils.model_utils import HuggingFaceModels
import numpy as np
import os
from functools import lru_cache
from typing import Dict, Tuple
import asyncio

# Set page config
st.set_page_config(
    page_title="Admin Dashboard - Banking Personalization System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for data management
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.interactions_df = None
    st.session_state.profiles_df = None
    st.session_state.products_df = None
    st.session_state.last_update = None

def load_data():
    """Load and preprocess data for the dashboard"""
    try:
        # Check if data is already loaded and not stale
        current_time = datetime.now()
        if (st.session_state.data_loaded and 
            st.session_state.last_update and 
            (current_time - st.session_state.last_update).total_seconds() < 300):  # 5-minute cache
            return (st.session_state.interactions_df, 
                   st.session_state.profiles_df, 
                   st.session_state.products_df)
        
        # Load data files
        interactions_df = pd.read_csv('data/user_interactions.csv')
        profiles_df = pd.read_csv('data/user_profiles.csv')
        products_df = pd.read_csv('data/financial_products.csv')
        
        # Convert timestamps using flexible format
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], format='mixed')
        if 'last_updated' in profiles_df.columns:
            profiles_df['last_updated'] = pd.to_datetime(profiles_df['last_updated'], format='mixed')
        
        # Process string lists
        for df in [interactions_df, products_df]:
            if 'features' in df.columns:
                df['features'] = df['features'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
            if 'tags' in df.columns:
                df['tags'] = df['tags'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
        
        # Process profile string lists
        for col in ['financial_goals', 'preferred_services']:
            if col in profiles_df.columns:
                profiles_df[col] = profiles_df[col].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
        
        # Update session state
        st.session_state.interactions_df = interactions_df
        st.session_state.profiles_df = profiles_df
        st.session_state.products_df = products_df
        st.session_state.data_loaded = True
        st.session_state.last_update = current_time
        
        return interactions_df, profiles_df, products_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def get_metrics(interactions_df: pd.DataFrame, profiles_df: pd.DataFrame) -> Dict:
    """Calculate key metrics"""
    try:
        # Basic metrics
        total_users = len(profiles_df['user_id'].unique())
        total_interactions = len(interactions_df)
        avg_feedback = interactions_df['feedback_score'].mean()
        
        # User interaction metrics
        total_clicks = interactions_df['clicked'].sum()
        total_selections = interactions_df['selected'].sum()
        click_rate = total_clicks / total_interactions if total_interactions > 0 else 0
        selection_rate = total_selections / total_interactions if total_interactions > 0 else 0
        
        # Feedback distribution
        feedback_distribution = {
            'high': len(interactions_df[interactions_df['feedback_score'] >= 0.7]),
            'medium': len(interactions_df[(interactions_df['feedback_score'] >= 0.4) & (interactions_df['feedback_score'] < 0.7)]),
            'low': len(interactions_df[interactions_df['feedback_score'] < 0.4])
        }
        
        return {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'avg_feedback': avg_feedback,
            'total_clicks': total_clicks,
            'total_selections': total_selections,
            'click_rate': click_rate,
            'selection_rate': selection_rate,
            'feedback_distribution': feedback_distribution
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def get_daily_metrics(interactions_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate daily metrics"""
    try:
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        daily_users = interactions_df.groupby(interactions_df['timestamp'].dt.date)['user_id'].nunique()
        daily_feedback = interactions_df.groupby(interactions_df['timestamp'].dt.date)['feedback_score'].mean()
        daily_clicks = interactions_df.groupby(interactions_df['timestamp'].dt.date)['clicked'].sum()
        daily_selections = interactions_df.groupby(interactions_df['timestamp'].dt.date)['selected'].sum()
        return daily_users, daily_feedback, daily_clicks, daily_selections
    except Exception as e:
        st.error(f"Error calculating daily metrics: {str(e)}")
        return pd.Series(), pd.Series(), pd.Series(), pd.Series()

def get_product_metrics(interactions_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate product metrics"""
    try:
        product_interactions = interactions_df['product_name'].value_counts()
        product_feedback = interactions_df.groupby('product_name')['feedback_score'].mean()
        product_clicks = interactions_df.groupby('product_name')['clicked'].sum()
        product_selections = interactions_df.groupby('product_name')['selected'].sum()
        return product_interactions, product_feedback, product_clicks, product_selections
    except Exception as e:
        st.error(f"Error calculating product metrics: {str(e)}")
        return pd.Series(), pd.Series(), pd.Series(), pd.Series()

def get_demographics(profiles_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Calculate demographic metrics"""
    try:
        income_dist = profiles_df['income_level'].value_counts()
        risk_dist = profiles_df['risk_tolerance'].value_counts()
        return income_dist, risk_dist
    except Exception as e:
        st.error(f"Error calculating demographics: {str(e)}")
        return pd.Series(), pd.Series()

def get_model_metrics() -> pd.DataFrame:
    """Load model metrics"""
    try:
        return pd.read_csv('models/model_metrics.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp', 'mse', 'r2', 'accuracy'])

def main():
    st.title("📊 Admin Dashboard")
    
    try:
        # Load data with progress indicator
        with st.spinner("Loading data..."):
            interactions_df, profiles_df, products_df = load_data()
        
        if interactions_df is None or profiles_df is None or products_df is None:
            st.error("Failed to load data. Please check the data files.")
            return
        
        # Sidebar filters
        with st.sidebar:
            st.header("Filters")
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                interactions_df = interactions_df[
                    (pd.to_datetime(interactions_df['timestamp']).dt.date >= start_date) &
                    (pd.to_datetime(interactions_df['timestamp']).dt.date <= end_date)
                ]
        
        # Display key metrics
        metrics = get_metrics(interactions_df, profiles_df)
        st.header("Key Metrics")
        
        # First row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", metrics['total_users'])
        with col2:
            st.metric("Total Interactions", metrics['total_interactions'])
        with col3:
            st.metric("Average Feedback Score", f"{metrics['avg_feedback']:.2f}")
        
        # Second row of metrics
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Total Clicks", metrics['total_clicks'])
        with col5:
            st.metric("Click Rate", f"{metrics['click_rate']:.2%}")
        with col6:
            st.metric("Selection Rate", f"{metrics['selection_rate']:.2%}")
        
        # User Engagement
        st.header("User Engagement")
        daily_users, daily_feedback, daily_clicks, daily_selections = get_daily_metrics(interactions_df)
        
        # Daily Active Users
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_users.index,
            y=daily_users.values,
            mode='lines+markers',
            name='Daily Active Users'
        ))
        fig1.update_layout(
            title='Daily Active Users',
            xaxis_title='Date',
            yaxis_title='Number of Users'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Daily Interaction Metrics
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily_clicks.index,
            y=daily_clicks.values,
            mode='lines+markers',
            name='Daily Clicks'
        ))
        fig2.add_trace(go.Scatter(
            x=daily_selections.index,
            y=daily_selections.values,
            mode='lines+markers',
            name='Daily Selections'
        ))
        fig2.update_layout(
            title='Daily Interaction Metrics',
            xaxis_title='Date',
            yaxis_title='Count'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Product Performance
        st.header("Product Performance")
        product_interactions, product_feedback, product_clicks, product_selections = get_product_metrics(interactions_df)
        
        # Product Interaction Metrics
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=product_interactions.index,
            y=product_interactions.values,
            name='Total Interactions'
        ))
        fig3.add_trace(go.Bar(
            x=product_clicks.index,
            y=product_clicks.values,
            name='Total Clicks'
        ))
        fig3.add_trace(go.Bar(
            x=product_selections.index,
            y=product_selections.values,
            name='Total Selections'
        ))
        fig3.update_layout(
            title='Product Interaction Metrics',
            xaxis_title='Product',
            yaxis_title='Count'
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Feedback Distribution
        st.subheader("Feedback Distribution")
        feedback_dist = metrics['feedback_distribution']
        fig4 = go.Figure()
        fig4.add_trace(go.Pie(
            labels=['High (≥0.7)', 'Medium (0.4-0.7)', 'Low (<0.4)'],
            values=[feedback_dist['high'], feedback_dist['medium'], feedback_dist['low']],
            name='Feedback Distribution'
        ))
        fig4.update_layout(title='Feedback Score Distribution')
        st.plotly_chart(fig4, use_container_width=True)
        
        # User Demographics
        st.header("User Demographics")
        income_dist, risk_dist = get_demographics(profiles_df)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Pie(
            labels=income_dist.index,
            values=income_dist.values,
            name='Income Level Distribution'
        ))
        fig5.update_layout(title='Income Level Distribution')
        st.plotly_chart(fig5, use_container_width=True)
        
        fig6 = go.Figure()
        fig6.add_trace(go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values,
            name='Risk Tolerance Distribution'
        ))
        fig6.update_layout(title='Risk Tolerance Distribution')
        st.plotly_chart(fig6, use_container_width=True)
        
        # Model Performance
        st.header("Model Performance")
        try:
            metrics_df = pd.read_csv('models/model_metrics.csv')
            
            if not metrics_df.empty:
                # Model Metrics
                col7, col8, col9 = st.columns(3)
                with col7:
                    latest_mse = metrics_df['mse'].iloc[-1]
                    st.metric("Latest MSE", f"{latest_mse:.4f}")
                with col8:
                    latest_r2 = metrics_df['r2'].iloc[-1]
                    st.metric("Latest R² Score", f"{latest_r2:.4f}")
                with col9:
                    latest_accuracy = metrics_df['accuracy'].iloc[-1] if 'accuracy' in metrics_df.columns else None
                    if latest_accuracy is not None:
                        st.metric("Latest Accuracy", f"{latest_accuracy:.4f}")
                
                # Model Performance Over Time
                fig7 = go.Figure()
                fig7.add_trace(go.Scatter(
                    x=pd.to_datetime(metrics_df['timestamp']),
                    y=metrics_df['mse'],
                    mode='lines+markers',
                    name='Mean Squared Error'
                ))
                fig7.update_layout(
                    title='Model MSE Over Time',
                    xaxis_title='Date',
                    yaxis_title='MSE'
                )
                st.plotly_chart(fig7, use_container_width=True)
                
                fig8 = go.Figure()
                fig8.add_trace(go.Scatter(
                    x=pd.to_datetime(metrics_df['timestamp']),
                    y=metrics_df['r2'],
                    mode='lines+markers',
                    name='R² Score'
                ))
                fig8.update_layout(
                    title='Model R² Score Over Time',
                    xaxis_title='Date',
                    yaxis_title='R² Score'
                )
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.warning("No model metrics available yet. Train the model to see performance metrics.")
        except FileNotFoundError:
            st.warning("No model metrics available yet. Train the model to see performance metrics.")
        
        # Raw Data Tables
        st.header("Raw Data")
        
        tab1, tab2 = st.tabs(["User Profiles", "Interactions"])
        
        with tab1:
            st.dataframe(profiles_df)
        
        with tab2:
            st.dataframe(interactions_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check if all required data files exist and are properly formatted.")

if __name__ == "__main__":
    main() 