import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2

def apply_oss(X, y):
    """
    Applies One-Sided Selection (OSS) for noise removal.
    1. 1-NN clustering to remove misclassified negative samples.
    2. Tomek Links removal.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
    """
    # 1. Start with initial set S (which is X, y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y  # Return as is if fully one-class

    # Initialize C with all positive instances and 1 random negative instance
    selected_neg_idx = np.random.choice(neg_idx, 1)
    C_idx = np.concatenate([pos_idx, selected_neg_idx])
    
    # Reclassify S using 1-NN trained on C
    X_C = X[C_idx]
    y_C = y[C_idx]
    
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(X_C, y_C)
    
    predictions = nn.predict(X)
    
    # Misclassified instances should be moved to C
    misclassified_idx = np.where(predictions != y)[0]
    C_idx = np.unique(np.concatenate([C_idx, misclassified_idx]))
    
    # Now find Tomek Links in C and remove the negative instances
    X_curr = X[C_idx]
    y_curr = y[C_idx]
    
    nn_tomek = KNeighborsClassifier(n_neighbors=2) # 1st is itself, 2nd is nearest neighbor
    nn_tomek.fit(X_curr, y_curr)
    
    # Get the nearest neighbor for each instance
    distances, indices = nn_tomek.kneighbors(X_curr)
    
    tomek_links = []
    for i in range(len(X_curr)):
        nn_index = indices[i][1] # Nearest neighbor (excluding itself)
        if indices[nn_index][1] == i and y_curr[i] != y_curr[nn_index]:
            # It's a Tomek Link! Note it down.
            if y_curr[i] == 0:
                tomek_links.append(i)
            elif y_curr[nn_index] == 0:
                tomek_links.append(nn_index)
                
    # Remove negative instances participating in Tomek Links
    final_idx = np.delete(np.arange(len(X_curr)), list(set(tomek_links)))
    
    X_clean = X_curr[final_idx]
    y_clean = y_curr[final_idx]
    
    return X_clean, y_clean


def apply_chi2_features(X, y, k=25):
    """
    Applies Chi-Squares feature selection to retrieve the top `k` features.
    Then pads or slices to strictly enforce k features.
    Reshapes output for CPN input (e.g., 25 -> 5x5).
    """
    # Ensure non-negative features for Chi2 (shifting by min value if necessary)
    X_shifted = X - np.min(X, axis=0)
    
    # Calculate Chi2 scores
    scores, p_values = chi2(X_shifted, y)
    
    # Sort features based on chi2 score descending
    sorted_indices = np.argsort(scores)[::-1]
    
    # Select top k
    selected_indices = sorted_indices[:k]
    
    if len(selected_indices) < k:
        # We need exactly k (25) features for the 5x5 structure. 
        # If the dataset has fewer features, we pad with zero columns.
        X_reduced = X[:, selected_indices]
        padding = np.zeros((X.shape[0], k - X_reduced.shape[1]))
        X_final = np.hstack([X_reduced, padding])
    else:
        X_final = X[:, selected_indices]
        
    return X_final, selected_indices
