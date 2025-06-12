import streamlit as st
import pandas as pd
from data_loader import DataLoader
from model import NewsClassifier
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fake {
        background-color: #ffcdd2;
        border: 1px solid #ef9a9a;
    }
    .real {
        background-color: #c8e6c9;
        border: 1px solid #81c784;
    }
    .confidence-high {
        color: #2e7d32;
    }
    .confidence-medium {
        color: #f57c00;
    }
    .confidence-low {
        color: #c62828;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📰 Fake News Detection System")
st.markdown("""
    This system uses machine learning to classify news articles as either FAKE or REAL.
    Enter a news article text below to get started.
""")

# Initialize session state for model
if 'model' not in st.session_state:
    if os.path.exists('model.joblib'):
        st.session_state.model = NewsClassifier()
        st.session_state.model.load_model()
    else:
        st.warning("Model not found. Please train the model first.")

# Input section
st.header("Input")
input_type = st.radio("Choose input type:", ["Single Article", "Multiple Articles"])

if input_type == "Single Article":
    text_input = st.text_area(
        "Enter news article text:",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    if st.button("Predict", type="primary"):
        if not text_input:
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                prediction, confidence = st.session_state.model.predict(text_input)
                
                # Display prediction
                st.header("Prediction")
                
                # Create prediction box with appropriate styling
                prediction_class = "fake" if prediction == "FAKE" else "real"
                confidence_class = (
                    "confidence-high" if confidence > 0.8
                    else "confidence-medium" if confidence > 0.6
                    else "confidence-low"
                )
                
                st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h2>Prediction: {prediction}</h2>
                        <p class="{confidence_class}">Confidence: {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display explanation
                st.markdown("""
                    ### How it works
                    This system uses natural language processing and machine learning to analyze the text and determine if it's likely to be fake or real news. The prediction is based on patterns learned from a large dataset of verified fake and real news articles.
                """)
else:
    uploaded_file = st.file_uploader("Upload a CSV file with news articles", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
        else:
            if st.button("Process Articles", type="primary"):
                with st.spinner("Processing articles..."):
                    results = []
                    for text in df['text']:
                        prediction, confidence = st.session_state.model.predict(text)
                        results.append({
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'prediction': prediction,
                            'confidence': confidence
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Show summary statistics
                    st.subheader("Summary Statistics")
                    st.write(f"Total articles processed: {len(results)}")
                    st.write(f"FAKE predictions: {sum(1 for r in results if r['prediction'] == 'FAKE')}")
                    st.write(f"REAL predictions: {sum(1 for r in results if r['prediction'] == 'REAL')}")
                    st.write(f"Average confidence: {sum(r['confidence'] for r in results) / len(results):.2%}")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
        This Fake News Detection System was developed as part of a Master's NLP assignment.
        
        ### Features:
        - Text preprocessing
        - TF-IDF vectorization
        - Multiple ML models
        - Hyperparameter tuning
        - Real-time prediction
        
        ### Model Information:
        - Best performing model: {model_name}
        - Features: TF-IDF with 10,000 features
        - Training data: Large dataset of fake and real news articles
    """.format(model_name=st.session_state.model.best_model_name if 'model' in st.session_state else "Not loaded"))
    
    st.markdown("""
        ### How to use:
        1. Enter a news article text or upload a CSV file
        2. Click 'Predict' or 'Process Articles'
        3. View the prediction and confidence score
        
        ### Confidence Levels:
        - High (>80%): Very confident prediction
        - Medium (60-80%): Moderately confident
        - Low (<60%): Less confident prediction
    """)

# Footer
st.markdown("---")
st.markdown("Developed for CT052-3-M-NLP Assignment | Task A - Text Classification")
st.markdown("Mohamed Sabiq Mohamed Sabry | TP085636") 