# TASK - 1
# EDA on retail sales data

import pandas as pd #helps us load, clean, analyze and summarize data
import seaborn as sns # creates meaningful graphs with less code
import matplotlib.pyplot as plt # turns data into visual graphs to help us understand trends and patterns

# Loading the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\menu.csv")

# checking if data is loaded correctly
df.head()
df.tail()
df.shape
df.columns

# Knowing the data structure
df.info()

# Handling missing values
df.isnull().sum()
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.duplicated().sum()

df. describe()

# Statistical summary of numerical columns
df['Category'].value_counts()
df.groupby('Category')['Calories'].mean().sort_values(ascending=False)
df[['Item', 'Calories']].sort_values(by='Calories', ascending=False).head(10)
df[['Item', 'Protein']].sort_values(by='Protein', ascending=False).head(10)

# Nutrient Correlation Analysis
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.show()

# Visualizations
# Average Calories per Category
df.groupby('Category')['Calories'].mean().plot(kind='bar')
plt.ylabel("Average Calories")
plt.title("Average Calories by Category")
plt.show()

# Sugar Distributation
sns.histplot(df['Sugars'], bins=20)
plt.title("Sugar Content Distribution")
plt.show()

# Fat vs Calories Scatter Plot
sns.scatterplot(x='Total Fat', y='Calories', data=df)
plt.show()
