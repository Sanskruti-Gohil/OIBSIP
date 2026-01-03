# TASK - 2
# CUSTOMER SEGMENTATION ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\Mall_Customers.csv")

# Rename columns for easier access
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income',
    'Spending Score (1-100)': 'Spending_Score'
}, inplace=True)

# Basic data inspection
print(df.info())
print(df.isnull().sum())

# Data cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Selecting relevant features
seg_data = df[['Annual_Income', 'Spending_Score']]

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(seg_data)

# Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Applying K-Means with optimal clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['Customer_Segment'] = kmeans.fit_predict(scaled_data)

# Visualizing customer segments
plt.figure()
sns.scatterplot(
    x=df['Annual_Income'],
    y=df['Spending_Score'],
    hue=df['Customer_Segment'],
    palette='Set2'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")
plt.show()

# Segment-wise analysis
print(df.groupby('Customer_Segment')[['Annual_Income', 'Spending_Score']].mean())
