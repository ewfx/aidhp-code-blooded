import streamlit as st
import pandas as pd
from utils.model_utils import HuggingFaceModels
from utils.user_interaction_handler import UserInteractionHandler
import json
import asyncio

async def get_recommendations_async(model_handler, user_id):
    return await model_handler.get_recommendations(user_id)

def show_recommendations_page():
    st.title("Personalized Financial Product Recommendations")
    
    # Initialize handlers
    model_handler = HuggingFaceModels()
    interaction_handler = UserInteractionHandler()
    
    # Get user ID from session state
    user_id = st.session_state.get('user_id')
    if not user_id:
        st.error("Please log in to view recommendations")
        return
    
    # Get user profile
    user_profile = interaction_handler.get_user_profile(user_id)
    if user_profile:
        st.subheader("Your Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Income Level:** {user_profile.get('income_level', 'Not specified')}")
            st.write(f"**Risk Tolerance:** {user_profile.get('risk_tolerance', 'Not specified')}")
        with col2:
            st.write(f"**Investment Goals:** {', '.join(user_profile.get('investment_goals', []))}")
            st.write(f"**Time Horizon:** {user_profile.get('time_horizon', 'Not specified')}")
        with col3:
            st.write(f"**Banking Frequency:** {user_profile.get('banking_frequency', 'Not specified')}")
            st.write(f"**Preferred Services:** {', '.join(user_profile.get('preferred_services', []))}")
    
    # Get recommendations using asyncio
    recommendations = asyncio.run(get_recommendations_async(model_handler, user_id))
    
    if not recommendations:
        st.info("No recommendations available at the moment. Please check back later.")
        return
    
    # Display recommendations with feedback
    st.subheader("Recommended Products")
    
    for rec in recommendations:
        with st.expander(f"**{rec['product_name']}** (Score: {rec['predicted_score']:.2f})"):
            # Display product details
            st.write(rec['description'])
            
            # Display complexity and risk analysis
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Complexity Analysis:**")
                st.write(f"- Average Sentence Length: {rec['complexity']['avg_sentence_length']:.2f}")
                st.write(f"- Complexity Score: {rec['complexity']['complexity_score']:.2f}")
                st.write(f"- Level: {rec['complexity']['complexity_level']}")
            
            with col2:
                st.write("**Risk Analysis:**")
                st.write(f"- Risk Score: {rec['risk_level']['risk_score']:.2f}")
                st.write(f"- Level: {rec['risk_level']['risk_level']}")
                st.write(f"- Sentiment: {rec['risk_level']['sentiment']['label']}")
            
            # Feedback section
            st.write("---")
            st.write("**Rate this recommendation:**")
            
            # Get feedback categories
            feedback_categories = model_handler.get_detailed_feedback()
            
            # Create feedback inputs
            feedback_data = {}
            for category, description in feedback_categories.items():
                st.write(f"**{description}**")
                feedback_data[category] = st.slider(
                    f"{category.capitalize()} Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key=f"{rec['product_name']}_{category}"
                )
            
            # Submit feedback button
            if st.button("Submit Feedback", key=f"submit_{rec['product_name']}"):
                if interaction_handler.collect_recommendation_feedback(user_id, rec['product_name'], feedback_data):
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Failed to submit feedback. Please try again.")
            
            # Show existing feedback if available
            if 'feedback' in rec and rec['feedback']['total_feedback'] > 0:
                st.write("---")
                st.write("**Previous Feedback:**")
                st.write(f"- Average Score: {rec['feedback']['average_score']:.2f}")
                st.write(f"- Total Feedback: {rec['feedback']['total_feedback']}")
    
    # Show feedback history
    st.subheader("Your Feedback History")
    feedback_history = interaction_handler.get_user_feedback(user_id)
    
    if feedback_history is not None and not feedback_history.empty:
        # Convert timestamp to datetime
        feedback_history['timestamp'] = pd.to_datetime(feedback_history['timestamp'])
        
        # Display feedback history
        for _, feedback in feedback_history.iterrows():
            with st.expander(f"{feedback['product_name']} - {feedback['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"Overall Score: {feedback['overall_score']:.2f}")
                for category in feedback_categories.keys():
                    st.write(f"{category.capitalize()}: {feedback[category]:.2f}")
    else:
        st.info("No feedback history available")

if __name__ == "__main__":
    show_recommendations_page() 