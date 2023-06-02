import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

# Define the OPTICS clustering algorithm
epsilon = 0.5  # Define the maximum distance between samples to be considered neighbors
min_samples = 2  # Define the minimum number of samples in a neighborhood for a point to be a core point
optics = OPTICS(eps=epsilon, min_samples=min_samples)

# Fit the OPTICS clustering algorithm to the data
optics.fit(X)  # use all columns for clustering

# Print the cluster labels for each data point
print(optics.labels_)

labels = optics.labels_
data['Label'] = labels

# Save the DataFrame with the labels to a new CSV file
data.to_csv('labeled_data_optics.csv', index=False)

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
plt.title('Scatter plot of Source and Label of labeled data')
plt.show()

# Plot graph
grouped = data.groupby(['Source']).size().reset_index(name='Counts')
plt.bar(grouped['Source'], grouped['Counts'])
plt.xlabel('Source')
plt.ylabel('Counts')
plt.title('Graph of labeled data')
plt.show()
