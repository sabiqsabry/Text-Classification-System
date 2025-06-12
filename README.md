# 📰 Fake News Classification System (Task A – CT052-3-M-NLP)

This project is a classical NLP-based text classification system built for **Task A of the CT052-3-M-NLP module** at **Asia Pacific University**. It detects whether a given news article is **FAKE** or **REAL** using supervised machine learning models and text preprocessing.

## 📂 Dataset

Dataset: [Fake News Detection Datasets – Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)  
Files used:
- `Fake.csv` – fake articles
- `True.csv` – real articles

Both were merged and labeled into a single training set with the `text` column used for prediction and a `label` column (`FAKE` or `REAL`).

## 🔧 Features

- Text preprocessing with lowercasing, punctuation removal, and optional lemmatization
- Feature extraction with `TfidfVectorizer`
- Supports 4 supervised ML models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Gradient Boosting
- Hyperparameter tuning using GridSearchCV
- Visual evaluation via:
  - Confusion matrices
  - Probability distribution plots
- Streamlit web app for live predictions

## 🚀 How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/sabiqsabry/Text-Classification-System.git
cd Text-Classification-System
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train.py
```

### 4. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 5. Paste a news article and click "Predict" to classify as FAKE or REAL.

## 📊 Model Evaluation

Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion matrix
- Probability distributions

## 🧠 Observations

| Model               | Accuracy  | Notes                    |
| ------------------- | --------- | ------------------------ |
| Gradient Boosting   | 🔥 A+     | Near-perfect performance |
| Random Forest       | 🔥 A+     | Very high precision      |
| Logistic Regression | ✅ Good    | Lightweight & reliable   |
| Naive Bayes         | 🟡 Decent | Prone to overconfidence  |

## 📁 Project Structure

| File               | Description                                        |
| ------------------ | -------------------------------------------------- |
| `data_loader.py`   | Loads and merges dataset                           |
| `model.py`         | Contains model training, tuning, evaluation        |
| `train.py`         | Script to train and save the model                 |
| `streamlit_app.py` | Streamlit GUI for interactive prediction           |
| `requirements.txt` | All Python dependencies                            |

## ⚠️ Notes

* This system does not use deep learning or transformer-based models
* All results are based on classical supervised ML as required by the assignment
* Dataset files (`Fake.csv` and `True.csv`) are not included in the repository due to size
* Generated model files and plots will be created during training

## ✍️ Author

Mohamed Sabiq Mohamed Sabry | TP085636
*Asia Pacific University – CT052-3-M-NLP*

## 📝 License

This project is part of an academic assignment and is not licensed for commercial use. 