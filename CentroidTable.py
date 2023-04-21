import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load preprocessed data
df = pd.read_csv("labeled_data2.csv")
df = df.drop(columns=["Time"])
# convert categorical variables into numerical variables
df['Source'] = pd.factorize(df['Source'])[0]
df['Destination'] = pd.factorize(df['Destination'])[0]
# df['Protocol'] = pd.factorize(df['Protocol'])[0]

# scale the features using min-max scaling
df = (df - df.min()) / (df.max() - df.min())

# apply k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
# get the coordinates of the cluster centers
centroids = kmeans.cluster_centers_

# plot the centroid table
# fig, ax = plt.subplots()
# ax.axis('off')
# ax.table(cellText=centroids, colLabels=df.columns, loc='center')
# plt.show()

# plot the data points and the centroids
plt.scatter(df['Source'], df['Destination'], c=kmeans.labels_, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('Source')
plt.ylabel('Destination')
plt.show()

# create a DataFrame from the centroids array
centroid_df = pd.DataFrame(centroids, columns=df.columns)

# save the DataFrame to a CSV file
centroid_df.to_csv('centroid_table.csv', index=False)