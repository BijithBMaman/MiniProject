import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# Load the preprocessed dataset into a Pandas DataFrame
df = pd.read_csv('data2.csv')
# df = df.drop(columns=["Source", "Destination"])

# Perform one-hot encoding on the 'Protocol' column
onehot_encoder = OneHotEncoder()
protocol_encoded = onehot_encoder.fit_transform(df[['Protocol']])
protocol_df = pd.DataFrame(protocol_encoded.toarray(), columns=onehot_encoder.get_feature_names(['Protocol']))
df = pd.concat([df, protocol_df], axis=1)
df = df.drop(['Protocol','Source','Destination'], axis=1)



# Perform k-means clustering on the dataset
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Print the cluster labels for each data point
print(labels)

# Add a new column to the DataFrame to store the labels
df['Label'] = labels

# Save the DataFrame with the labels to a new CSV file
df.to_csv('labeled_data.csv', index=False)