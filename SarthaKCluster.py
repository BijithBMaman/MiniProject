import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into a pandas DataFrame
data = pd.read_csv(r'C:\Users\Public\My Project\Miniproject\MiniProject\NewTest\MixTrafficPreprocess.csv')

# Define the categorical and numerical columns
cat_cols = ['Source', 'Destination', 'Protocol']
num_cols = ['Time', 'Length']

# Define the column transformer
ct = ColumnTransformer([
    ('encoder', OneHotEncoder(sparse=False), cat_cols),
    ('scaler', StandardScaler(), num_cols)
])

# Transform the data
X = ct.fit_transform(data)

# Define the k-means clustering algorithm
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit the k-means clustering algorithm to the data
kmeans.fit(X) # only use the first column (Time) for clustering

# Print the cluster labels for each data point
print(kmeans.labels_)

labels = kmeans.labels_
data['Label'] = labels

# Save the DataFrame with the labels to a new CSV file
data.to_csv('labeled_data2.csv', index=False)

# Plot scatter plot of Soure and time of labeled data
plt.scatter(data['Source'], data['Time'])
plt.xlabel('Source')
plt.ylabel('Time')
plt.title('Scatter plot of Source and time of labeled data')
plt.show()

# Plot scatter plot of Soure and label of labeled data
plt.scatter(data['Source'], data['Label'])
plt.xlabel('Source')
plt.ylabel('Label')
plt.title('Scatter plot of Soure and Label of labeled data')
plt.show()

# Plot graph
grouped = data.groupby(['Source']).size().reset_index(name='Counts')
plt.bar(grouped['Source'], grouped['Counts'])
plt.xlabel('Source')
plt.ylabel('Counts')
plt.title('Graph of labeled data')
plt.show()


# # Get the cluster centroids
# centroids = kmeans.cluster_centers_

# # Create a DataFrame of the cluster centroids with the correct column names
# cat_col_names = ct.named_transformers_['encoder'].get_feature_names(cat_cols)
# col_names = list(cat_col_names) + num_cols
# centroid_df = pd.DataFrame(centroids, columns=col_names)

# # Plot a heatmap of the cluster centroids
# sns.heatmap(centroid_df, cmap="YlGnBu")
# plt.title('Heatmap of Cluster Centroids')
# plt.show()