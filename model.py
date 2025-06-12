import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, classification_report
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class NewsClassifier:
    def __init__(self):
        """
        Initialize the NewsClassifier with vectorizer and models.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.9
        )
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': CalibratedClassifierCV(MultinomialNB(), cv=5),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, df):
        """
        Prepare data for training by vectorizing text.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame with 'cleaned_text' column
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['label']
        
        # Use stratified split to maintain label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Log split statistics
        logging.info("\nTraining set label distribution:")
        logging.info(y_train.value_counts(normalize=True).round(3))
        logging.info("\nTest set label distribution:")
        logging.info(y_test.value_counts(normalize=True).round(3))
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
        """
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, pos_label='FAKE'),
                'recall': recall_score(y_test, y_pred, pos_label='FAKE'),
                'f1': f1_score(y_test, y_pred, pos_label='FAKE')
            }
            
            results[name] = metrics
            
            # Log detailed results
            logging.info(f"\nResults for {name}:")
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            
            for metric, value in metrics.items():
                logging.info(f"{metric.capitalize()}: {value:.4f}")
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name}.png')
            plt.close()
            
            # Plot probability distribution
            plt.figure(figsize=(10, 6))
            for label in ['FAKE', 'REAL']:
                mask = y_test == label
                plt.hist(y_pred_proba[mask, 0], bins=50, alpha=0.5, label=label)
            plt.title(f'Probability Distribution - {name}')
            plt.xlabel('Probability of FAKE')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(f'probability_dist_{name}.png')
            plt.close()
        
        # Find best model based on F1 score
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logging.info(f"\nBest model: {best_model_name}")
        
        return results
    
    def tune_hyperparameters(self, X_train, y_train):
        """
        Perform hyperparameter tuning on the best model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Create a custom F1 scorer with the correct pos_label
        f1_scorer = make_scorer(f1_score, pos_label='FAKE')
        
        if self.best_model_name == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': [None, 'balanced']
            }
        elif self.best_model_name == 'naive_bayes':
            param_grid = {
                'base_estimator__alpha': [0.1, 0.5, 1.0]
            }
        elif self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': [None, 'balanced']
            }
        else:  # gradient_boosting
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        
        grid_search = GridSearchCV(
            self.models[self.best_model_name],
            param_grid,
            cv=5,
            scoring=f1_scorer,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logging.info(f"\nBest parameters for {self.best_model_name}:")
        logging.info(grid_search.best_params_)
        
        self.best_model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def predict(self, text):
        """
        Make prediction for new text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            tuple: (prediction, probability)
        """
        # Vectorize the input text
        X = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.best_model.predict(X)[0]
        
        # Get probability scores
        probabilities = self.best_model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, path='model.joblib'):
        """
        Save the trained model and vectorizer.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.best_model,
            'model_name': self.best_model_name
        }
        joblib.dump(model_data, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path='model.joblib'):
        """
        Load a trained model and vectorizer.
        
        Args:
            path (str): Path to the saved model
        """
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        logging.info(f"Model loaded from {path}") 