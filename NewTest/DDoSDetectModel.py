import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\Public\My Project\Miniproject\MiniProject\NewTest\MixTrafficPreprocess.csv")

# Perform one-hot encoding for categorical features
data_encoded = pd.get_dummies(data, columns=["Source", "Destination", "Protocol"])

# Select relevant features
features = ["Time", "Length"]
X = data_encoded[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
epsilon = 0.5  # Define the maximum distance between samples to be considered neighbors
min_samples = 5  # Define the minimum number of samples in a neighborhood for a point to be a core point
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

# Access cluster labels
cluster_labels = dbscan.labels_

# Count the number of clusters (excluding noise)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
num_noise_points = list(cluster_labels).count(-1)

# Print the number of clusters and noise points
print("Number of clusters:", num_clusters)
print("Number of noise points:", num_noise_points)

# Plot the clusters
plt.scatter(X["Time"], X["Length"], c=clusters)
plt.xlabel("Time")
plt.ylabel("Length")
plt.title("DBSCAN Clustering")
plt.show()
