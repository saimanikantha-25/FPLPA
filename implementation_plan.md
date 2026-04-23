# FPLPA Framework Implementation

This architecture outlines how we will construct the Federated Prototype Learning based on Prototype Averaging (FPLPA) system for Cross-Project Defect Prediction.

## User Review Required
> [!IMPORTANT]
> Please review this structure below. It uses PyTorch as the primary deep learning framework. Additionally, `scikit-learn` and `pandas` will be utilized. If these are not installed, we should install them first via `pip`.

## Proposed Changes

We will create the following Python files in the current workspace to ensure modularity and clean code.

---

### Preprocessing Module
#### [NEW] [data_processing.py](file:///c:/Users/saima/OneDrive/Desktop/miniprj/data_processing.py)
This script will house the code for **Phase 1 (Data Preprocessing)**.
- **`apply_oss(X, y)`**: Logic for One-Sided Selection (using `KNeighborsClassifier` and finding/removing Tomek links from negative samples).
- **`apply_chi2_features(X, y, k=25)`**: Applies the Chi-Squares test to select the most correlated 25 features, then pads/reshapes them into the necessary `(1, 5, 5)` tensor for the Convolutional Prototype Network.

---

### Local Model Definition (CPN)
#### [NEW] [models.py](file:///c:/Users/saima/OneDrive/Desktop/miniprj/models.py)
This module establishes **Phase 2 (Local Training Model)**.
- **`CPN(nn.Module)`**: A Convolutional Prototype Network architecture.
  - Convolutional + Max Pooling feature extractor.
  - Embeds input into an embedding vector `v_i(φ_i; x)`.
  - Defines custom feed-forward logic for yielding prediction labels AND embedding vectors necessary for generating prototype centroids.

---

### Federated Flow Handlers
#### [NEW] [client.py](file:///c:/Users/saima/OneDrive/Desktop/miniprj/client.py)
- **`FPLClient` Class**: Responsible for a single project's local training loop. Computes Supervised Error vs labels + the Regularization Mean Squared Error between its local prototype and the Server's Global Prototype.

#### [NEW] [server.py](file:///c:/Users/saima/OneDrive/Desktop/miniprj/server.py)
This module handles **Phase 3 (Prototype Aggregation)**.
- **`FPLServer` Class**: Functions as the central hub. Aggregates the local prototypes. Simulates the optimal random subsetting strategy (e.g., pulling 3 random local prototypes per interaction to smooth variance) and broadcasts the new global averages.

---

### Driver Entry Point & Utilities
#### [NEW] [utils.py](file:///c:/Users/saima/OneDrive/Desktop/miniprj/utils.py)
- Functions for loading data from `.csv` files stored under the `AEEEM`, `NASA`, and `Relink` folders.
- Evaluation metrics implementation (`get_auc`, `get_g_mean`).

#### [NEW] [main_federated.py](file:///c:/Users/saima/OneDrive/Desktop/miniprj/main_federated.py)
- Replicates **Phase 4 (Training loop)**. 
- Loads the datasets, initializes multiple instances of `FPLClient`, one instance of `FPLServer`, and orchestrates iterative updates across federated communication rounds. Conducts the final testing utilizing Euclidean distance against global prototypes.

## Open Questions

> [!WARNING]
> 1. Which Python environment are you using, and are `torch`, `torchvision`, `scikit-learn`, `numpy`, and `pandas` already installed? If not, I can create a `requirements.txt` and install them.
> 2. For testing, do you want to start by building all these files first, or start strictly with testing Phase 1 (Data Preprocessing logic) so we can be sure it processes your CSV files correctly?
> 3. Does the first training test run need to focus on a specific combination of sets right away (e.g., AEEEM + NASA), or a mock combination first for debugging?

## Verification Plan

### Automated Tests
- I'll run `python main_federated.py` manually utilizing a small slice of mock data (or one tiny CSV, e.g. `mw1.csv` + another dataset) to trace and ensure gradients form, PyTorch compiles without shape mismatch errors, and Server/Client communication remains stable across epochs.

### Manual Verification
- We will inspect the console outputs to ensure that local prototype vectors are not raw data leaks, conforming strictly to an aggregated float tensor shape.
