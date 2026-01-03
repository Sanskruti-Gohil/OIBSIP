# Task - 8
# Unveiling the Android App Market: Analyzing Google Play Store Data

import pandas as pd

# Loading the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\datasets\\apps.csv")

df.head()
df.shape
df.info()
df.isnull().sum()

df.dropna(subset=['Rating'], inplace=True)

df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)

df['Price'] = df['Price'].str.replace(r'\$', '', regex=True).astype(float)

def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return None

df['Size'] = df['Size'].astype(str).apply(convert_size)

df['Category'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.countplot(y='Category', data=df)
plt.title("App Distribution Across Categories")
plt.show()

df['Rating'].describe()

sns.histplot(df['Rating'], bins=20)
plt.title("Distribution of App Ratings")
plt.show()

sns.boxplot(x='Type', y='Installs', data=df)
plt.title("Installs: Free vs Paid Apps")
plt.show()

sns.scatterplot(x='Size', y='Rating', data=df)
plt.title("App Size vs Rating")
plt.show()

sns.histplot(df[df['Type'] == 'Paid']['Price'])
plt.title("Price Distribution of Paid Apps")
plt.show()

df['Sentiment'] = df['Rating'].apply(
    lambda x: 'Positive' if x >= 4 else 'Neutral' if x >= 3 else 'Negative'
)

sns.countplot(x='Sentiment', data=df)
plt.title("User Sentiment Based on Ratings")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].corr(), cmap='coolwarm')
plt.title("Correlation Between App Metrics")
plt.show()

