# Principal Component Analysis (PCA) Implementation

This repository contains a Python implementation of Principal Component Analysis (PCA) for dimensionality reduction and variance analysis. PCA is a powerful statistical technique used to identify patterns in data by transforming it into a set of orthogonal (uncorrelated) components, ranked by the amount of variance they explain.

## Project Overview

This project demonstrates the following key steps in performing PCA:

1. **Loading and Centering Data**: Loading the dataset and centering it by subtracting the mean.
2. **Computing the Covariance Matrix**: Calculating the covariance matrix of the centered data.
3. **Eigenvalue and Eigenvector Computation**: Computing the eigenvalues and eigenvectors of the covariance matrix to identify the principal components.
4. **Variance Analysis**: Plotting the cumulative variance explained by the principal components to determine the number of components needed to retain a desired amount of variance.
5. **Visualization**: Visualizing the first two principal components and the reduced-dimension data.
6. **Comparison with scikit-learn PCA**: Comparing the custom implementation with the PCA implementation from scikit-learn.

## Features

- **Data Centering**: Centering the data to have a mean of zero.
- **Covariance Matrix Calculation**: Efficient computation of the covariance matrix.
- **Eigendecomposition**: Extraction and sorting of eigenvalues and eigenvectors.
- **Variance Explained Plot**: Visualization of the cumulative variance explained by the principal components.
- **Principal Components Plot**: Visualization of the first two principal components.
- **Reduced-Dimension Data Plot**: Scatter plot of the data in the reduced-dimension space.
- **scikit-learn Comparison**: Using scikit-learn's PCA for comparison and validation.

## Technologies Used

- **Python**
- **NumPy**
- **Matplotlib**
- **scikit-learn**

## Getting Started

### Prerequisites

Ensure you have Python and the following libraries installed:

- NumPy
- Matplotlib
- scikit-learn

You can install the required libraries using pip:

```bash
pip install numpy matplotlib scikit-learn
