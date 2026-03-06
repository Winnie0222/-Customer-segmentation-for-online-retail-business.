Customer Segmentation using RFM Analysis and Clustering
------------------------------------------------------------

Project Overview
This project focuses on segmenting customers based on their purchasing behavior using RFM (Recency, Frequency, Monetary) analysis. The goal is to identify distinct customer segments to support data-driven marketing strategies, personalized promotions, and business decision-making.

Dataset
The dataset is obtained at Kaggle (https://www.kaggle.com/datasets/ulrikthygepedersen/online-retail-dataset)
The dataset contains historical customer transactions, including:
- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- Customer ID
- Country

The data is preprocessed to calculate RFM features for each customer:
- Recency: Days since last purchase
- Frequency: Number of transactions
- Monetary: Total purchase value

Methodology
RFM Feature Engineering
Calculated Recency, Frequency, and Monetary scores.
Normalized data for clustering algorithms.

Clustering Approaches
K-Means Clustering: Used Elbow Method and Silhouette Score to determine optimal number of clusters.
DBSCAN: Density-based clustering to detect noise and outliers.
Hierarchical Clustering: Agglomerative clustering to observe hierarchical structure of customer segments.

Evaluation & Visualization
Compared clusters using silhouette scores and visual inspection.
Visualized customer segments using 2D projections and heatmaps.
Interpreted segments to provide actionable business insights.

Technologies Used
Python (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy)
Jupyter Notebook for interactive analysis

Key Features
Compute RFM scores to quantify customer value
Apply multiple clustering algorithms for robust segmentation:
- K-Means (with Elbow Method & Silhouette Score)
- DBSCAN
- Hierarchical Clustering
Compare clusters using metrics like Silhouette Score
Visualize and interpret customer segments for actionable business insights
Interactive Deployment with Streamlit: Users can input RFM values and segment customers dynamically using a smooth and intuitive interface


Authors
Kam Win Ni



