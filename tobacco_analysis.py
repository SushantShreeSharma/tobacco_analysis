import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("smokers.csv")

# Handling missing values
df.fillna(df.median(), inplace=True)

# Feature Engineering
df['Smoking_Duration'] = df['Age'] - df['Age_Started_Smoking']

# Encoding categorical variables
encoder = OneHotEncoder()
df_encoded = pd.get_dummies(df, columns=['Gender', 'Smoking_Status'])

# Data Splitting
X = df_encoded.drop(columns=['Smoking_Status'])
y = df_encoded['Smoking_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = rf_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
feature_importance = rf_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10,5))
sns.barplot(x=feature_importance[sorted_idx], y=feature_names[sorted_idx], palette="coolwarm")
plt.title("Feature Importance in Smoking Prediction")
plt.show()
