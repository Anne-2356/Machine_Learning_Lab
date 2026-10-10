import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic.csv")

# Display first 5 rows
df.head()

# Shape of dataset
print("Rows and Columns:", df.shape)

# Column names
print(df.columns)

# Information about dataset
df.info()

# Summary statistics
df.describe()

# Summary including categorical columns
df.describe(include='all')

# Missing values
df.isnull().sum()

# Number of duplicate rows
df.duplicated().sum()

# Remove duplicates
df = df.drop_duplicates()

df.nunique()

df['Embarked'].unique()

df.dtypes

pd.crosstab(df['Sex'], df['Survived'])

pd.crosstab(df['Pclass'], df['Survived'])

## UNIVARIANT ANALYSIS
## HIST
df.hist(figsize=(12,8))
plt.show()

##HISTOGRAM
plt.figure(figsize=(6,4))
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

##BOX PLOT
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Age'])
plt.title("Boxplot of Age")
plt.show()

##COUNT PLOT
sns.countplot(x='Sex', data=df)
plt.show()

##PIE CHART
df['Survived'].value_counts().plot(kind='pie',
                                   autopct='%1.1f%%')
plt.title("Survival Distribution")
plt.show()

##BIVARIATE ANALYSIS
##SCATTER PLOT
plt.scatter(df['Age'], df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

##LINE PLOT
sns.lineplot(x='Pclass', y='Fare', data=df)
plt.show()

##BAR PLOT
sns.barplot(x='Pclass', y='Fare', data=df)
plt.show()

##BOX PLOT
sns.boxplot(x='Survived', y='Age', data=df)
plt.show()

##COUNT PLOT
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

##CORRELARION ANALYSIS
##CORRELATION MATRIX
numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

corr

##HEAT MAP
plt.figure(figsize=(8,6))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')
plt.show()

## OUTLIER DETECTION
sns.boxplot(data=df[['Age','Fare']])
plt.show()

df.groupby('Pclass')['Fare'].mean()

df.groupby('Sex')['Age'].mean()

df.groupby('Survived')['Fare'].mean()
