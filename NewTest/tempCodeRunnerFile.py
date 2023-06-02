import numpy as np
from sklearn.cluster import KMeans

# Step 1: Preprocess the data
# Assume you have converted the IP addresses to numerical representations

# Step 2: Extract features
data = np.array([
    [10, 2, 1, 74],
    [10, 2, 5, 66],
    [10, 2, 5, 66],
    [10, 2, 5, 66],
    # ... include other data points from your dataset
])

# Step 4: Assign labels
labels = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Step 5: Train the K-means model
k = 2  # Number of clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(data)

# Step 6: Get cluster labels for the data points
cluster_labels = model.labels_

print(cluster_labels)