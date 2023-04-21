import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('data1.csv')

# Check for missing values in the DataFrame
missing_values = df.isnull().sum()

# Print the missing values
print(missing_values)

df = pd.read_csv('data1.csv', na_values=['missing'])
print(df)
