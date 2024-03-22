# SMS Spam Detection Project

## Overview
This project aims to classify SMS messages as either spam or ham (non-spam) using machine learning techniques. The dataset used for this project contains SMS messages labeled as spam or ham.

## Project Structure
- **Data Cleaning**: Initially, the dataset is loaded and cleaned by removing unnecessary columns, handling missing values, and removing duplicates.
- **Exploratory Data Analysis (EDA)**: This section involves exploring the dataset to understand its characteristics, including visualizations of spam vs. ham distribution, word clouds, and descriptive statistics.
- **Data Preprocessing**: Text data preprocessing steps are performed, including lowercasing, tokenization, removing special characters, stopwords, and punctuation, as well as stemming.
- **Model Building**: Various machine learning models are trained on the preprocessed data, including Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Support Vector Classifier, Decision Tree Classifier, Random Forest Classifier, AdaBoost Classifier, Bagging Classifier, Gradient Boosting Classifier, and XGBoost Classifier. Model performance metrics such as accuracy and precision are evaluated.
- **Model Improvement**: Strategies for improving model performance are explored, including limiting the number of features during vectorization, scaling, and ensemble techniques such as Voting Classifier and Stacking Classifier.
- **Final Model Selection**: The best performing model is selected based on performance metrics and is saved for future use.
- **Deployment**: The final model and vectorizer are saved using pickle for deployment in production environments.

## Files
- **spam.csv**: The dataset containing SMS messages labeled as spam or ham.
- **model.pkl**: Pickle file containing the trained model (Multinomial Naive Bayes).
- **vectorizer.pkl**: Pickle file containing the vectorizer used for text preprocessing.

## Instructions for Usage
1. Install the required libraries using `pip install -r requirements.txt`.
2. Run the Jupyter Notebook `sms_spam_detection.ipynb` to train the model and perform analysis.
3. Use the trained model and vectorizer for classification tasks in other applications.

## Dependencies
- Python 3.x
- Libraries: numpy, pandas, scikit-learn, nltk, matplotlib, seaborn, wordcloud, xgboost

## Acknowledgements
- The dataset used in this project was obtained from-- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- First look of the dataset obtained using pandas-profiling-- file:///C:/Users/Dell/SMS%20spam%20detection/output.html
## Author
- Sheersh Saxena
