import logging
from data_loader import DataLoader
from model import NewsClassifier

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize data loader
    data_loader = DataLoader(
        fake_path='News _dataset/Fake.csv',
        true_path='News _dataset/True.csv'
    )
    
    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    df = data_loader.load_and_merge_data()
    df = data_loader.preprocess_data(df)
    
    # Initialize classifier
    classifier = NewsClassifier()
    
    # Prepare data for training
    logging.info("Preparing data for training...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Train and evaluate models
    logging.info("Training and evaluating models...")
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Perform hyperparameter tuning
    logging.info("Performing hyperparameter tuning...")
    best_params = classifier.tune_hyperparameters(X_train, y_train)
    
    # Save the model
    logging.info("Saving the model...")
    classifier.save_model()
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main() 