# pca_transform.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

def analyze_dna_sequences(human_texts, y_data):
    """
    Analyzes DNA sequences using PCA and visualizes the results.

    Parameters:
    - human_texts: List of DNA sequence strings (e.g., k-mers).
    - y_data: List of class labels corresponding to the sequences.
    """
    # Step 1: Feature Extraction
    cv = CountVectorizer(ngram_range=(1, 4))  # Adjusted to allow k-mers of length 1 to 4
    X = cv.fit_transform(human_texts).toarray()

    # Step 2: Standardize the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Perform PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_pca = pca.fit_transform(X_scaled)

    # Step 4: Visualize PCA Results
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_data, cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of DNA Sequences')
    plt.colorbar(label='Class')
    plt.show()