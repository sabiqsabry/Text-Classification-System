import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, fake_path, true_path):
        """
        Initialize DataLoader with paths to fake and true news datasets.
        
        Args:
            fake_path (str): Path to Fake.csv
            true_path (str): Path to True.csv
        """
        self.fake_path = fake_path
        self.true_path = true_path
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add common fake news phrases to stop words
        fake_news_phrases = {
            'click here', 'breaking', 'shocking', 'you won\'t believe',
            'must read', 'viral', 'exclusive', 'urgent', 'breaking news',
            'just in', 'latest', 'update', 'developing', 'watch now',
            'share this', 'sponsored', 'advertisement'
        }
        self.stop_words.update(fake_news_phrases)
        
    def load_and_merge_data(self):
        """
        Load and merge fake and true news datasets.
        
        Returns:
            pd.DataFrame: Merged and shuffled DataFrame
        """
        # Load datasets
        fake_df = pd.read_csv(self.fake_path)
        true_df = pd.read_csv(self.true_path)
        
        # Add labels
        fake_df['label'] = 'FAKE'
        true_df['label'] = 'REAL'
        
        # Merge datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log dataset info
        logging.info("\nDataset Statistics:")
        logging.info(f"Total number of articles: {len(df)}")
        logging.info("\nLabel distribution:")
        logging.info(df['label'].value_counts())
        logging.info("\nLabel distribution percentages:")
        logging.info(df['label'].value_counts(normalize=True).round(3))
        
        return df
    
    def clean_text(self, text):
        """
        Clean and preprocess text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove dates (various formats)
        text = re.sub(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', '', text)
        text = re.sub(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', '', text)
        
        # Remove author names (common patterns)
        text = re.sub(r'by\s+[\w\s]+$', '', text)
        text = re.sub(r'written by\s+[\w\s]+$', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_data(self, df, text_column='text'):
        """
        Preprocess the entire dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Column name containing the text to process
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Drop unused columns
        columns_to_keep = [text_column, 'label']
        df = df[columns_to_keep].copy()
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Log text length statistics
        df['text_length'] = df['cleaned_text'].str.len()
        logging.info("\nText length statistics by label:")
        logging.info(df.groupby('label')['text_length'].describe())
        
        logging.info("Text preprocessing completed")
        
        return df 