# Banking Personalization System

A sophisticated banking personalization system that provides personalized financial product recommendations using advanced NLP and machine learning techniques.

## Features

### 1. User Management
- Secure user authentication (login/signup)
- User profile management
- Session management
- Password hashing and security

### 2. Personalized Recommendations
- AI-powered product recommendations
- BERT-based text analysis for product descriptions
- Complexity analysis of financial products
- Risk level assessment
- Sentiment analysis of product descriptions

### 3. User Preferences
- Income level selection
- Risk tolerance assessment
- Financial goals setting
- Preferred services selection
- Banking frequency preferences

### 4. Feedback System
- Detailed product feedback collection
- Multi-category rating system
- Feedback history tracking
- Real-time feedback processing

### 5. Admin Dashboard
- User engagement metrics
- Model performance monitoring
- Product popularity analysis
- User demographics visualization
- Feedback analytics

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.12
- **Machine Learning**: 
  - Hugging Face Transformers (BERT)
  - scikit-learn
  - pandas
  - numpy
- **Data Storage**: CSV files
- **Authentication**: Custom user management system

## Project Structure

```
.
├── app.py                 # Main application file
├── admin_dashboard.py     # Admin dashboard
├── train_model.py         # Model training script
├── data/                  # Data directory
│   ├── financial_products.csv
│   ├── user_profiles.csv
│   └── user_interactions.csv
├── pages/                 # Streamlit pages
│   └── recommendations.py # Recommendations page
├── utils/                 # Utility modules
│   ├── data_processor.py
│   ├── model_utils.py
│   ├── user_interaction_handler.py
│   └── user_manager.py
└── models/               # Saved models directory
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd code
   cd src
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory with:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

5. **Initialize Data Files**
   - Ensure the `data` directory contains:
     - `financial_products.csv`
     - `user_profiles.csv`
     - `user_interactions.csv`

6. **Train the Model**
   ```bash
   python train_model.py
   ```

## Running the Application

1. **Start the Main Application**
   ```bash
   streamlit run app.py
   ```

2. **Access the Admin Dashboard**
   ```bash
   streamlit run admin_dashboard.py
   ```

## Usage Guide

1. **User Registration/Login**
   - Create a new account or login with existing credentials
   - Set up your financial preferences

2. **Viewing Recommendations**
   - Access personalized product recommendations
   - View detailed product information
   - Provide feedback on recommendations

3. **Admin Dashboard**
   - Monitor system performance
   - View user engagement metrics
   - Analyze feedback data

## Data Structure

### User Profiles
- User ID
- Income Level
- Risk Tolerance
- Financial Goals
- Preferred Services
- Banking Frequency
- Last Updated

### Financial Products
- Product Name
- Description
- Features
- Tags
- Minimum Balance
- Interest Rate
- Annual Fee
- Monthly Fee
- Minimum Investment
- Management Fee
- Tax Benefits

### User Interactions
- User ID
- Product Name
- Timestamp
- Interaction Type
- Feedback Score
- Clicked
- Selected

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing the BERT model
- Streamlit for the web framework
- All contributors and maintainers 