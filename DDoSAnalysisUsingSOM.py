import pandas as pd
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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

# Define the SOM parameters
n_rows = 10
n_cols = 10
input_len = X.shape[1]
sigma = 1.0
learning_rate = 0.5
iterations = 100

# Initialize the SOM
som = MiniSom(n_rows, n_cols, input_len, sigma=sigma, learning_rate=learning_rate)

# Train the SOM
som.random_weights_init(X)
som.train_random(X, iterations)

# Get the coordinates of the winning neurons for each input sample
winning_neurons = np.array([som.winner(x) for x in X])

# Map the winning neuron coordinates to cluster labels
labels = np.ravel_multi_index(winning_neurons.T, (n_rows, n_cols)) # type: ignore

# Assign the labels to the data
data['Label'] = labels

# Save the DataFrame with the labels to a new CSV file
data.to_csv('labeled_data_som.csv', index=False)

# Plot scatter plot of Source and time of labeled data
plt.scatter(data['Source'], data['Time'])
plt.xlabel('Source')
plt.ylabel('Time')
plt.title('Scatter plot of Source and time of labeled data')
plt.show()

# Plot scatter plot of Source and label of labeled data
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
