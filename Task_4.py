# TASK - 4
# Cleaning Data

import pandas as pd
# Loading the data
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\Datasets\\AB_NYC_2019.csv")

df.head()
df.shape
df.info()

df.describe()

df.isnull().sum()

df['name'].fillna('Not Available', inplace=True)
df['host_name'].fillna('Unknown', inplace=True)
df['last_review'] = pd.to_datetime(df['last_review'])
df['reviews_per_month'].fillna(0, inplace=True)

df.duplicated().sum() # Check for duplicate rows
df.drop_duplicates(inplace=True)

# Standardizing column names
df['neighbourhood_group'] = df['neighbourhood_group'].str.lower().str.strip()
df['room_type'] = df['room_type'].str.lower().str.strip()
df['last_review'] = pd.to_datetime(df['last_review'])
df = df[df['price'] > 0]

# Detecting outliers
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['price'] >= Q1 - 1.5 * IQR) &
        (df['price'] <= Q3 + 1.5 * IQR)]

df.info()
df.isnull().sum()
df.duplicated().sum()

# Saving the cleaned data
df.to_csv("AB_NYC_2019_CLEANED.csv", index=False)
df.head()