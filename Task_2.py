# TASK - 2
# CUSTOMER SEGEMENT ANALYSIS

import pandas as pd

# Load the dataset
df=pd.read_csv("C:/Users/spgoh/Downloads/archive (1)/Mall_Customers.csv")

# View the data
df.head()
df.shape
df.columns

# Check for datatypes and null values
df.info()
df.isnull().sum()

df.dropna(inplace=True) # Handling missing values
df.drop_duplicates(inplace=True) # removing duplicate values
seg_data = df[['Annual Income (k$)', 'Spending Score (1-100)']] # selecting useful columns
seg_data.describe()

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(seg_data)

import matplotlib.pyplot as plt

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Customer Segmentation using K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df['Customer_Segment'] = clusters

# Visualizing of customer segments
import seaborn as sns

sns.scatterplot(
    x=df['Annual_Income'],
    y=df['Spending_Score'],
    hue=df['Customer_Segment'],
    palette='Set2'
)
plt.title("Customer Segmentation")
plt.show()

df.groupby('Customer_Segment').mean()