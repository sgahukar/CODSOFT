# Spam SMS Detection 📩

## Overview  
This project classifies SMS messages as spam or not using text classification models. It implements a **Spam Classification** model using **TF-IDF vectorization** and a **Naive Bayes classifier** to distinguish between spam and legitimate messages.

## Dataset  
- [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- The dataset contains SMS messages labeled as `ham` (legitimate) or `spam`.
- The dataset is loaded from `spam.csv`.

## Steps in the Project
1. **Data Cleaning**
   - Selecting relevant columns (`label` and `message`).
   - Mapping labels (`ham` → 0, `spam` → 1).
   - Checking and handling missing values.
2. **Exploratory Data Analysis**
   - Generating a **missing values heatmap**.
3. **Feature Engineering**
   - Using **TF-IDF Vectorization** to transform text into numerical format.
4. **Model Training**
   - Training a **Multinomial Naive Bayes** classifier.
5. **Model Evaluation**
   - Measuring **accuracy, precision, recall, and F1-score**.

## Requirements
Install the required dependencies:
```bash
pip install pandas scikit-learn seaborn matplotlib

