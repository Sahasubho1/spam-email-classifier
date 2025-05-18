# Spam Email Classifier

A simple spam email classifier built using traditional machine learning algorithms.

## Features

- Text preprocessing with lemmatization (NLTK)
- TF-IDF vectorization
- Logistic Regression with class balancing
- Confusion matrix visualization (Seaborn)
- Pipeline serialization using `joblib`

## Libraries

- Scikit-learn
- NLTK
- Seaborn
- Matplotlib
- Pandas
- Numpy


## Description

The model was trained and evaluated using the [Spam Email Dataset](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) from Kaggle. This dataset comprises approximately 87% non-spam (ham)(label: 0) emails and 13% spam(label: 1) emails. To address the class imbalance, class weights were applied during model training.

For preprocessing, each email message was lemmatized using the WordNet Lemmatizer from the NLTK library. Text data was then vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer instead of a Count Vectorizer. TF-IDF was chosen because it reduces the weight of commonly occurring words and emphasizes rare but potentially important words—often more indicative of spam.

For classification, we used Logistic Regression, which outperformed other models such as Random Forest, AdaBoostClassifier, and Decision Tree Classifier. Class weights were passed directly into the Logistic Regression model to further handle the imbalance.

The preprocessing steps and classifier were integrated using a Scikit-learn pipeline to prevent data leakage and ensure a smooth training and evaluation workflow. The final model achieved a test accuracy of 98.12%.
## Installation

```bash
git clone https://github.com/Sahasubho1/spam-email-classifier.git
cd spam-email-classifier
pip install -r requirements.txt
```

## Usage

You can use the saved model (`classifier.pkl`) to make predictions on new email text.

```python
import joblib
model = joblib.load("classifier.pkl")
prediction = model.predict(["Free money!!! Click here now!"])
print(prediction)
```






