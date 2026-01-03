# Task - 6
# Wine Quality Prediction

import pandas as pd

# Loading the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\WineQT.csv")

df.head()
df.shape
df.info()
df.describe()

df.isnull().sum()
df.drop_duplicates(inplace=True)

X = df.drop('quality', axis=1)
y = df['quality']

y = y.apply(lambda x: 1 if x >= 7 else 0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("SGD Accuracy:", accuracy_score(y_test, sgd.predict(X_test)))
print("SVC Accuracy:", accuracy_score(y_test, svc.predict(X_test)))

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y)
plt.title("Wine Quality Distribution")
plt.show()

import numpy as np

importances = rf.feature_importances_
feature_names = X.columns

sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance for Wine Quality")
plt.show()