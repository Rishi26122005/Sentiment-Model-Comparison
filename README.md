# IMDB Sentiment Analysis & Rating Prediction

## 📌 Project Overview
This project applies Natural Language Processing (NLP) and Machine Learning techniques to classify IMDB movie reviews. The goal is to predict whether a movie's rating is **High** (above average) or **Low** (below average) based solely on the text content of the review.

The project compares multiple supervised learning algorithms against a lexicon-based baseline (TextBlob) to determine the most effective approach for sentiment classification.

## 📂 Dataset
The dataset (`IMBD_movie_rating.csv`) consists of movie reviews and metadata.
- **Input:** `Review` (Text data)
- **Target:** `rate_label` (Binary classification)
  - **1:** Global Rating > Average Rating (~5.73)
  - **0:** Global Rating ≤ Average Rating

## 🛠️ Methodology

### 1. Data Preprocessing
- **Data Cleaning:** Removed unnecessary columns (Title, Runtime, Dates, etc.) and handled null values.
- **Text Normalization:**
  - Converted text to lowercase.
  - Removed URLs, Twitter handles (`@`), hashtags (`#`), and special characters using Regex.
  - Removed extra whitespace.
- **Label Creation:** Converted numerical ratings into binary class labels based on the dataset's mean rating.

### 2. Feature Extraction
- **TF-IDF Vectorization:** Used `TfidfVectorizer` to convert text into numerical vectors.
  - *N-grams:* Unigrams and Bigrams (1,2)
  - *Max Features:* 15,000
  - *Stop Words:* English

### 3. Models Implemented
The following models were trained and evaluated:
1. **Multinomial Naive Bayes** (Baseline for text data)
2. **Random Forest Classifier** (Ensemble method)
3. **Logistic Regression** (Linear model)
4. **Support Vector Machine (SVM)** (Linear Kernel)
5. **TextBlob** (Unsupervised Lexicon-based sentiment analysis for comparison)

## 📊 Results
The models were evaluated based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**. The SVM model achieved the highest performance.

| Model | Accuracy | F1-Score (Positive Class) |
| :--- | :---: | :---: |
| **SVM (Linear Kernel)** | **73.39%** | **0.75** |
| Logistic Regression | 73.20% | 0.76 |
| Naive Bayes | 72.96% | 0.75 |
| Random Forest | 69.51% | 0.73 |
| TextBlob (Baseline) | 60.71% | 0.70 |

### ROC Curve Analysis
The notebook includes visualizations of the **Receiver Operating Characteristic (ROC)** curves and **AUC** (Area Under Curve) scores to evaluate the trade-off between sensitivity and specificity for each model.

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib textblob
