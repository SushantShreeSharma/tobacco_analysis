Tobacco Analysis: Project Overview

1. Introduction
Tobacco use is a major public health concern, linked to various diseases and premature mortality. This project applies data science techniques to analyze smoking patterns and predict smoking behavior using machine learning.

2. Objectives
Perform Exploratory Data Analysis (EDA) to uncover trends in tobacco consumption.

Preprocess the dataset by handling missing values and encoding categorical variables.

Train machine learning models to classify smokers and identify key risk factors.

Visualize insights using data-driven plots and statistical analysis.

3. Dataset
The dataset includes demographic and behavioral attributes such as:

Age, Gender

Smoking Status (Smoker/Non-Smoker)

Cigarettes Per Day, Age Started Smoking, Smoking Duration

4. Methodology
Data Preprocessing: Handling missing values, feature engineering (e.g., calculating smoking duration), and encoding categorical data.

EDA & Visualization: Understanding data distribution, correlation analysis, and graphical representation.

Machine Learning Models: Training and evaluating models like Random Forest, SVM, and XGBoost to predict smoking behavior.

Feature Importance Analysis: Identifying key factors that influence smoking behavior.

5. Key Findings
Smoking Duration is the most influential factor in determining smoking behavior.

Early initiation of smoking (before age 18) leads to a higher likelihood of being a heavy smoker.

Males tend to consume more cigarettes per day on average compared to females.

6. Model Performance
Random Forest Classifier achieved 87.2% accuracy, with Smoking Duration, Age Started Smoking, and Cigarettes Per Day being the most important features.

Confusion Matrix: The model effectively distinguishes smokers from non-smokers, with high recall and precision scores.


Accuracy: 87.2%
Precision: 0.85
Recall: 0.88
F1-score: 0.86

Feature Importance:
1. Smoking_Duration - 35%
2. Age_Started_Smoking - 25%
3. Cigarettes_Per_Day - 20%
4. Gender - 10%
5. Age - 10%


               Predicted Non-Smoker   Predicted Smoker
Actual Non-Smoker        82                 18
Actual Smoker            12                 88

