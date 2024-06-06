import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# (a) Load the data into a Python program and center it
data = np.loadtxt('cloud (1).csv', delimiter=',')
data_centered = data - np.mean(data, axis=0)
print(data_centered)

# (b) Compute the covariance matrix Σ
cov_matrix = np.dot(data_centered.T, data_centered) / data.shape[0]

#Compute eigendecomposition
values, vectors = np.linalg.eig(cov_matrix)

# (c) Compute the eigenvectors and eigenvalues of Σ
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# (d) Plot the percentage of retained variance
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Principal Components')
plt.grid(True)
plt.show()

num_components_for_90_variance = np.argmax(cumulative_variance_ratio >= 0.90) + 1
print("Number of PCs for 90% retained variance:", num_components_for_90_variance)

# (e) Plot the first two components (eigenvectors)
plt.plot(range(1, len(eigenvectors) + 1), eigenvectors[:, 0], label='PC1')
plt.plot(range(1, len(eigenvectors) + 1), eigenvectors[:, 1], label='PC2')
plt.xlabel('Dimension')
plt.ylabel('Eigenvector Magnitude')
plt.legend()
plt.title('Eigenvectors for PC1 and PC2')
plt.grid(True)
plt.show()

# (f) Compute the reduced-dimension data matrix D2 and plot it
reduced_data = np.dot(data_centered, eigenvectors[:, :2])

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter Plot of Reduced-Dimension Data')
plt.grid(True)
plt.show()

#  (g) Study the PCA implementation in scikit-learn
pca = PCA(n_components=num_components_for_90_variance)
pca.fit(data_centered)
sklearn_eigenvalues = pca.explained_variance_
print("Eigenvalues from scikit-learn PCA:", sklearn_eigenvalues)






