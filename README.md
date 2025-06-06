# elevatelabs-task8
# 🧠 Mall Customer Segmentation using K-Means Clustering

## 📍 Project Overview

This project applies **K-Means Clustering** to segment mall customers based on:
- **Age**
- **Annual Income (k$)**
- **Spending Score (1–100)**

By identifying patterns in customer behavior, businesses can offer more targeted and efficient services or marketing strategies.

---

## 🧾 Dataset

- **File:** `Mall_Customers.csv`
- **Source:** [Kaggle – Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Features Used:**
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)

---

## Dependencies

| Tool           | Purpose                              |
|----------------|--------------------------------------|
| `pandas`       | Data loading and manipulation        |
| `numpy`        | Numerical operations                 |
| `matplotlib`   | Data visualization                   |
| `scikit-learn` | K-Means, StandardScaler, PCA, Metrics|

---

## 🔄 Workflow Summary

### 📌 Step-by-Step

1. **Load Dataset:**  
   Read and inspect the dataset using `pandas`.

2. **Feature Selection:**  
   Used `Age`, `Annual Income`, and `Spending Score`.

3. **Data Preprocessing:**  
   Standardize features using `StandardScaler` to ensure equal contribution.

4. **Dimensionality Reduction (PCA):**  
   Reduce to 2 components for 2D visualization.

5. **Determine Optimal K:**  
   - **Elbow Method:** Plots inertia vs number of clusters.
   - **Silhouette Score:** Measures cluster quality.
   
6. **Train K-Means Model:**  
   Fit and predict clusters for the chosen value of K (e.g., K=6).

7. **Assign Cluster Labels:**  
   Add predicted cluster labels to the original dataset.

8. **Visualize Clusters (2D PCA):**  
   Plot clusters in 2D space using `matplotlib`.

9. **Evaluate Performance:**  
   Use Silhouette Score to evaluate cluster separation.

---

## 📈 Outputs

### ✅ Elbow Plot (Inertia vs K)

> Helps identify the “elbow point” where adding more clusters doesn’t significantly reduce inertia.

![Elbow Plot](https://i.imgur.com/yRblK2k.png)

---

### ✅ Silhouette Score Plot

> Shows which value of K gives the best-defined clusters.

![Silhouette Plot](https://i.imgur.com/zSgXHdA.png)

---

### ✅ PCA Cluster Visualization

> Visualizes customer clusters in 2D after dimensionality reduction.

![Cluster Visualization](https://i.imgur.com/BsY1VGg.png)

---

## 📊 Results

- **Optimal K (Clusters):** 6  
- **Silhouette Score:** 0.4311 (Good separation between clusters)  
- **Insights:**  
  - High-spending customers form clear groups  
  - Some clusters correspond to young high-income customers  
  - Others represent low-income low-spending segments

---

## 📌 Business Implications

- 📦 Target promotions to specific customer segments.
- 🛒 Understand high-value customers' behavior.
- 🎯 Allocate marketing budget effectively.

---

## 🧑‍💻 How to Run the Code

```bash
# 1. Install required libraries
pip install pandas numpy matplotlib scikit-learn

# 2. Place 'Mall_Customers.csv' in the same directory

# 3. Run the script or Jupyter notebook
python kmeans_clustering.py  # or run notebook cells
