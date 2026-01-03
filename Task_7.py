# Task - 7
# Credit Card Fraud Detection

import pandas as pd

# Loading the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\creditcard.csv")

df.head()
df.shape
df.info()

df['Class'].value_counts()

df.isnull().sum()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.001, random_state=42)
df['anomaly'] = iso.fit_predict(df.drop('Class', axis=1))

X = df.drop('Class', axis=1)
y = df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

from sklearn.metrics import classification_report

print(classification_report(y_test, lr.predict(X_test)))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = lr.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fraud Detection")
plt.show()