# Task - 3
# Sentiment Analysis on twitter data 

import pandas as pd

# Loading the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\Twitter_Data.csv")

df.head()
df.shape
df.info()

# Rename column
df.rename(columns={'category': 'sentiment'}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

df['sentiment'].value_counts()  # Checking the distribution of sentiment labels

# preprocessing the data
# NOTE: Dataset already contains 'clean_text'
# NLTK is NOT required here, so we safely disable it

import re
# import nltk
# from nltk.corpus import stopwords

# stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    # words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# Train test split
from sklearn.model_selection import train_test_split

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

# Evaluating matrics
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Data Visualization
# Sentiment distribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

# Word frequency vizualization
from wordcloud import WordCloud

text = " ".join(df['clean_text'])
wordcloud = WordCloud(background_color='white').generate(text)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()
