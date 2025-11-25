# iFond - Repository Details 

This repository contains a Python implementation of the iFond (Information Fact Checker from Online News Data) framework. The code follows the three-principal steps outlined in the research paper:

1.  **Data Preparation (Imputation)**
2.  **Selection of Significant Variables (Dimensionality Reduction)**
3.  **Application of Data-Driven Models (Classification-Association Rules)**

## Getting Started

### Prerequisites

You need a standard Python environment with the following libraries:

```bash
pip install numpy pandas scikit-learn
```

## File Structure

The entire framework is contained within a single Python file, typically named `ifond_framework.py`.

## How to Run the Code

The main execution pipeline is included in the `ifond_framework.py` file under the `if __name__ == '__main__':` block, running on a set of dummy text and numerical data.

1.  Save the code as `ifond_framework.py`.
2.  Run the file from your terminal:

<!-- end list -->

```bash
python ifond_framework.py
```

### Expected Output

The script will print the results of the structural process, including the shape changes after feature engineering, imputation, and dimensionality reduction, followed by the final classification results based on the placeholder association rules.

```
==============================================
iFond Hybrid Misinformation Detection Framework (Expanded Structural)
==============================================
--- 0. Performing Text Feature Engineering (Bag-of-Words) ---
Generated feature matrix with shape: (5, 14)
--- 1. Performing CMVE-like Imputation (KNN-Substitute, k=2) ---
Imputation completed. Initial NaNs: 2, Final NaNs: 0
--- 2. Performing DR (PCA-Substitute, n_components=2) ---
DR completed. Original shape: (5, 14), Reduced shape: (5, 2)
--- 3. Training iFond Classification-Association Rules Classifier ---
Generated 2 IF-THEN Classification Rules (Structural implementation).
Final Classification (Structural Test):
True Labels:  [0 1 0 0 1]
Pred Labels:  [0 1 0 0 1]
Accuracy: 1.00 (Note: Based on structural substitutes)
==============================================
```

## Framework Components

The framework is implemented using modular functions and classes as below.

### 0\. Feature Engineering

  * **Function:** `text_to_feature_matrix(texts)`
  * **Purpose:** Converts raw news text into a numerical feature matrix.
  * **Implementation:** Uses a simplified **Bag-of-Words** approach to count word occurrences.

### 1\. Data Preparation (Imputation)

  * **Class:** `CMVEEstimator(n_neighbors=5)`
  * **Goal:** Implement **Collateral Missing Values Imputation (CMVE)** using covariance and ranking to select effective rows ($D_k$).
  * **Notes:** Uses **K-Nearest Neighbors (KNN) logic** for estimation. This method structurally matches the concept of estimating missing values based on the proximity (similarity) of other data points, a core idea in CMVE.

### 2\. Dimensionality Reduction (DR)

  * **Class:** `BinaryRankOneDR(n_components=5)`
  * **Goal:** Implement a custom DR technique based on **binary rank-one approximation** ($A=xy^T$) and iterative partitioning to select the most significant variables.
  * **Notes:** Implements **Principal Component Analysis (PCA)** using manual Singular Value Decomposition (SVD) logic. PCA is a standard, robust technique for finding lower-dimensional representations that capture maximum variance, serving as a functional equivalent for compression.

### 3\. Classification-Association Rules (CAR)

  * **Class:** `IFondCARClassifier(min_support=0.1, min_confidence=0.7)`
  * **Goal:** Utilize a modified **Apriori algorithm** to generate high-confidence Classification-Association Rules (CAR) where the consequence is the target class (Fake/True).
  * **Notes:** Contains a simplified, structural implementation of the **standard Apriori algorithm** to generate frequent itemsets and subsequent IF-THEN rules, demonstrating the process of association rule mining for classification.

## 🛠️ Paper
```
@INPROCEEDINGS{11154637,
  author={Ramachandran, Chandrasekar},
  booktitle={2025 IEEE/ACIS 23rd International Conference on Software Engineering Research, Management and Applications (SERA)}, 
  title={Misinformation detection using hybrid statistical and machine learning techniques}, 
  year={2025},
  volume={},
  number={},
  pages={175-178},
  keywords={Accuracy;Systematics;Social networking (online);Machine learning;Organizations;Medical services;Predictive models;Internet;Fake news;Software engineering;data mining;machine learning},
  doi={10.1109/SERA65747.2025.11154637}}
```
