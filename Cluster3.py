import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load data into a pandas DataFrame
df = pd.read_csv('data1.csv')

# drop any rows with missing values
df.dropna(inplace=True)

# convert categorical variables into numerical variables
df['Source'] = pd.factorize(df['Source'])[0]
df['Destination'] = pd.factorize(df['Destination'])[0]
df['Protocol'] = pd.factorize(df['Protocol'])[0]

# scale the features using min-max scaling
df = (df - df.min()) / (df.max() - df.min())

# apply k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

# add a new column to the DataFrame indicating which cluster each row belongs to
df['cluster'] = kmeans.labels_

# plot a scatter plot of the clusters separately
for i in range(2):
    cluster_df = df[df['cluster'] == i]
    plt.scatter(cluster_df['Source'], cluster_df['Destination'], s=10, label=f'Cluster {i}')
plt.xlabel('Source')
plt.ylabel('Destination')
plt.legend()
plt.show()
