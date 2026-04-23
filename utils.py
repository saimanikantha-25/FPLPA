import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import math

def safe_load_csv(filepath):
    """ Loads CSV safely, handling potential bad encoding/formats. """
    try:
        df = pd.read_csv(filepath)
    except Exception:
        # Fallback reading
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    return df

def extract_features_and_labels(df):
    """
    Assumes last column is target. 
    Binarizes target variable to 1 (True/Defective) and 0 (False/Clean).
    """
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    
    y = np.zeros(len(y_raw), dtype=int)
    for i, val in enumerate(y_raw):
        val_str = str(val).strip().lower()
        if val_str in ['true', 'y', '1', 'yes', 'defective', 'buggy', 't']:
            y[i] = 1
        elif val_str in ['false', 'n', '0', 'no', 'clean', 'non-defective', 'f']:
            y[i] = 0
        else:
            try:
                # Direct cast if numerical but stored awkwardly
                num = float(val)
                y[i] = 1 if num > 0 else 0
            except:
                y[i] = 0
                
    # Cast X to float just in case
    X = X.astype(np.float64)
    
    # Handle NaN values explicitly
    X = np.nan_to_num(X)
    
    return X, y

def load_dataset(base_dir, project_name):
    """
    Loads all CSVs found in base_dir/project_name and consolidates them into X, y.
    Returns None, None if the folder doesn't exist.
    """
    target_path = os.path.join(base_dir, project_name)
    if not os.path.exists(target_path):
        return None, None
        
    X_list = []
    y_list = []
    
    for fname in os.listdir(target_path):
        if fname.endswith(".csv"):
            df = safe_load_csv(os.path.join(target_path, fname))
            X, y = extract_features_and_labels(df)
            X_list.append(X)
            y_list.append(y)
            
    if not X_list:
        return None, None
        
    return np.vstack(X_list), np.concatenate(y_list)

def get_auc(y_true, y_pred):
    """ Returns AUC given flat arrays """
    # Convert tensors to numpy if they aren't already
    if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'): y_pred = y_pred.cpu().numpy()
    
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Happens if only one class is present in true labels
        return 0.5 

def get_g_mean(y_true, y_pred):
    """ Returns G-mean given flat arrays """
    if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'): y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return math.sqrt(sensitivity * specificity)
        
    return 0.0 # Invalid confusion matrix shape (e.g. only 1 class predicted/true)
