import pandas as pd

# Read in the CSV file
df = pd.read_csv("MixedTraffic.csv")

# Specify the source IPs that you want to keep as a list
source_ips_to_keep = ['10.0.0.1','10.0.0.2','10.0.0.3', '10.0.0.4', '10.0.0.5']

# Filter the DataFrame to keep only rows with the specified source IPs
df = df[df['Source'].isin(source_ips_to_keep)]

# Reset the index of the filtered DataFrame
df.reset_index(drop=True, inplace=True)

# Drop the unwanted columns
df = df.drop_duplicates()
df = df.dropna()
df = df.drop(columns=["No.", "Info"])

# Write the result to a new CSV file
df.to_csv("MixTrafficPreprocess.csv", index=False)