# Iris Flower Classification

## Project Overview

This project is developed as part of the Syntecxhub Machine Learning Internship (Task 2).

The project uses the Iris dataset to classify flowers into three species:
- Setosa
- Versicolor
- Virginica

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Steps

### 1. Exploratory Data Analysis
- Loaded the Iris dataset.
- Created a Pandas DataFrame.
- Checked dataset shape and information.
- Generated statistical summary.
- Checked for missing values.
- Visualized feature pairs using a pair plot.

### 2. Model Training

Two classification models were trained:

- Logistic Regression
- Decision Tree Classifier

### 3. Model Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.67% |
| Decision Tree | 93.33% |

Logistic Regression performed better on the test dataset.

### 4. Confusion Matrix

The confusion matrix for Logistic Regression showed:

- Setosa: 10 correctly classified
- Versicolor: 9 correctly classified, 1 misclassified as Virginica
- Virginica: 10 correctly classified

Only one flower was misclassified.

### 5. New Flower Prediction

A simple command-line interface was created to allow the user to enter measurements of a new flower:

- Sepal length
- Sepal width
- Petal length
- Petal width

The trained Logistic Regression model then predicts the flower species.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt