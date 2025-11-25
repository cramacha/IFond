import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin

# ==============================================================================
# 0. Feature Engineering (Preprocessing for Text Data)
# ==============================================================================

def text_to_feature_matrix(texts: List[str]) -> np.ndarray:
    """
    Simulates the process of extracting features (like n-grams/TF-IDF) .
    
    A simple Bag-of-Words approach is used here.
    """
    print("--- 0. Performing Text Feature Engineering (Bag-of-Words) ---")
    
    # 1. Build Vocabulary
    word_counts = defaultdict(int)
    for text in texts:
        for word in text.lower().split():
            word_counts[word] += 1
            
    # Filter common words and assign indices
    vocabulary = {word: i for i, (word, count) in enumerate(word_counts.items()) if count > 1}
    vocab_size = len(vocabulary)
    
    # 2. Create Feature Matrix (Count Vectorizer)
    feature_matrix = np.zeros((len(texts), vocab_size))
    
    for i, text in enumerate(texts):
        for word in text.lower().split():
            if word in vocabulary:
                feature_matrix[i, vocabulary[word]] += 1
                
    print(f"Generated feature matrix with shape: {feature_matrix.shape}")
    return feature_matrix

# ==============================================================================
# 1. Data Preparation: Collateral Missing Values Imputation (CMVE-like)
# ==============================================================================

class CMVEEstimator(BaseEstimator):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        print(f"--- 1. Performing CMVE-like Imputation (KNN-Substitute, k={self.n_neighbors}) ---")
        X_imputed = X.copy()
        
        # 8. Repeat the previously stated steps until all missing values are imputed.
        missing_mask = np.isnan(X_imputed)
        if not np.any(missing_mask):
            return X_imputed

        # Simple iterative process
        for i, j in np.argwhere(missing_mask):
            # Isolate the current row and column for the missing value
            X_imputed[i, j] = np.nanmean(X[:, j])
            # -----------------------------------------------

        print(f"Imputation completed. Initial NaNs: {np.sum(missing_mask)}, Final NaNs: {np.sum(np.isnan(X_imputed))}")
        return X_imputed

# ==============================================================================
# 2. Dimensionality Reduction (Binary Rank-One Approximation)
# ==============================================================================

class BinaryRankOneDR(BaseEstimator):
    
    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        print(f"\n--- 2. Performing DR (PCA-Substitute, n_components={self.n_components}) ---")
        
        # Ensure the input X is binarized or discretized first.

        m, n = X.shape
        if min(m, n) < 2:
            return X

  
        X_centered = X - np.mean(X, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False) 
        
        # Calculate eigenvalues and eigenvectors
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues (descending)
        idx = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, idx]
        
        # Select n_components
        self.W = eigen_vectors[:, :self.n_components]
        
        # Transform data: Output is a k_i-complete set of data points D_{k_i}
        X_reduced = X_centered @ self.W
        # -----------------------------------------------------------------

        print(f"DR completed. Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")
        return X_reduced

# ==============================================================================
# 3. Classification-Association Rules (CAR) Model
# ==============================================================================

class IFondCARClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules: List[Tuple] = []

    def _generate_frequent_itemsets(self, transactions: np.ndarray) -> List[Tuple]:
        """Simplified structural Apriori Algorithm."""
        frequent_itemsets = []
        num_transactions = len(transactions)
        
        # 1. Generate L1 (Frequent 1-itemsets)
        L_k = {}
        for item_idx in range(transactions.shape[1]):
            support = np.sum(transactions[:, item_idx]) / num_transactions
            if support >= self.min_support:
                L_k[frozenset([item_idx])] = support
        
        frequent_itemsets.append(L_k)
        
        k = 2
        # 2. While C_k is not empty
        while frequent_itemsets[-1]:
            C_k = {} # Candidate k-itemsets
            L_prev = frequent_itemsets[-1]

            # a. Construct C_k from L_{k-1}
            for itemset1 in L_prev.keys():
                for itemset2 in L_prev.keys():
                    union = itemset1.union(itemset2)
                    if len(union) == k:
                        C_k[union] = 0

            # c. Check the support for all itemsets in C_k (Scan transactions)
            for itemset in C_k.keys():
                count = 0
                for transaction in transactions:
                    # Check if all items in the itemset are present in the transaction
                    if all(transaction[i] == 1 for i in itemset):
                        count += 1
                C_k[itemset] = count / num_transactions

            # d. Eliminate itemsets that fail to meet the minimum support criteria
            L_k = {itemset: support for itemset, support in C_k.items() if support >= self.min_support}
            
            if L_k:
                frequent_itemsets.append(L_k)
            else:
                break
            k += 1

        # Flatten the list of dictionaries into a single list of (itemset, support)
        flat_list = []
        for d in frequent_itemsets:
            for itemset, support in d.items():
                flat_list.append((itemset, support))
        return flat_list

    def _generate_classification_rules(self, frequent_itemsets: List[Tuple], y_transactions: np.ndarray) -> List[Tuple]:
        """
        Generates classification rules (IF-THEN) where the target T is on the RHS.
        """
        rules = []
        target_col = y_transactions.shape[1] - 1
        
        for itemset, support in frequent_itemsets:
            if target_col in itemset:
                # The itemset contains the target class (T)
                
                # Rule is: (itemset - {T}) => {T}
                antecedent = itemset.difference({target_col})
                consequent = frozenset([target_col])
                
                if not antecedent: continue # Skip if rule is just {} => {T}

                confidence = support / self.min_support # Simple placeholder for confidence
                
                if confidence >= self.min_confidence:
                    rules.append((antecedent, consequent, confidence))
                # ----------------------------------------------------
                    
        return rules

    def fit(self, X: np.ndarray, y: np.ndarray):
        
        # Ensure y is a column vector and concatenate with X
        data = np.hstack([X, y.reshape(-1, 1)])

        # Data must be binarized/discrete for Apriori
        # NOTE: Assuming X (from DR) and y are already binarized/discretized.
        
        # 1-2. Apply Apriori-like algorithm (Simplified structural Apriori)
        frequent_itemsets = self._generate_frequent_itemsets(data)
        
        # 3. Generate Classification Rules (IF-THEN rules)
        self.rules = self._generate_classification_rules(frequent_itemsets, data)
        
        print(f"Generated {len(self.rules)} IF-THEN Classification Rules (Structural implementation).")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts classes based on the generated highest-confidence rules."""
        predictions = []
        for row in X:
            best_match_class = 0 # Default to class 0
            max_confidence = -1
            
            for antecedent, consequent, confidence in self.rules:
                # Check if the row satisfies the antecedent (IF part)
                antecedent_met = all(row[idx] == 1 for idx in antecedent)
                
                if antecedent_met and confidence > max_confidence:
                    max_confidence = confidence
                    # The consequent is the target class (T), which is the last column index
                    best_match_class = list(consequent)[0] 
                    
            predictions.append(best_match_class)
            
        return np.array(predictions)

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
def run_ifond_framework_full():
    """Coordinates the full three-step process on dummy news data."""
    print("==============================================")
    print("iFond Hybrid Misinformation Detection Framework (Expanded Structural)")
    print("==============================================")
    
    # ------------------- Create Dummy Data -------------------
    raw_articles = [
        "vaccines can prevent coronavirus",
        "covid-19 vaccines contain microchips tracking",
        "coronavirus originated from laboratories",
        "new vaccine approved for all ages", 
        "chips are found in all covid tests"
    ]
    # Target: 0=True/Neutral, 1=Fake/False
    raw_labels = np.array([0, 1, 0, 0, 1]) 
    
    # ------------------- 0. Feature Engineering -------------------
    X_features = text_to_feature_matrix(raw_articles)
    
    # Introduce some NaNs for imputation test
    X_features[0, 2] = np.nan
    X_features[4, 5] = np.nan
    
    # ------------------- 1. Imputation -------------------
    imputer = CMVEEstimator(n_neighbors=2)
    X_imputed = imputer.fit_transform(X_features)
    
    # ------------------- 2. Dimensionality Reduction -------------------
    # The output must be discretized for the final CAR step.
    dr = BinaryRankOneDR(n_components=2) 
    X_reduced_cont = dr.fit_transform(X_imputed)
    
    # Discretization: Must be performed before CAR
    # We binarize the reduced features based on their mean for a simple stub.
    X_discretized = (X_reduced_cont > np.mean(X_reduced_cont, axis=0)).astype(int)

    # ------------------- 3. CAR Classification -------------------
    # The target y must also be included in the Apriori transactions.
    classifier = IFondCARClassifier(min_support=0.4, min_confidence=0.7)
    classifier.fit(X_discretized, raw_labels)
    
    # Predict and evaluate
    y_pred = classifier.predict(X_discretized)
    accuracy = np.mean(y_pred == raw_labels)
    
    print(f"\nFinal Classification (Structural Test):")
    print(f"True Labels:  {raw_labels}")
    print(f"Pred Labels:  {y_pred}")
    print(f"Accuracy: {accuracy:.2f} (Note: Based on structural substitutes)")
    print("==============================================")

if __name__ == '__main__':
    run_ifond_framework_full()
