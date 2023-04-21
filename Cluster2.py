import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the preprocessed dataset into a Pandas DataFrame
df = pd.read_csv('data2.csv')

# Perform one-hot encoding on the 'Protocol' column
onehot_encoder = OneHotEncoder()
protocol_encoded = onehot_encoder.fit_transform(df[['Protocol']])
protocol_df = pd.DataFrame(protocol_encoded.toarray(), columns=onehot_encoder.get_feature_names(['Protocol']))
df = pd.concat([df, protocol_df], axis=1)
df = df.drop(['Protocol','Source','Destination'], axis=1)

#
# Perform k-means clustering on the dataset
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Plot the data points with different colors based on their cluster labels
# plt = plt.plot(labels)
plt.scatter(df['Length'], df['Protocol_BitTorrent'], c=labels)
plt.xlabel('Source')
plt.ylabel('Destination')
plt.show()
