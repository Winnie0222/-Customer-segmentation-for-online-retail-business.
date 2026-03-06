import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title='Customer Segmentation',
    page_icon=':bar_chart:',  # This is an emoji shortcode. Could be a URL too.
)

st.title('Customer Segmentation')

# Load your dataset with caching
@st.cache_data
def load_data():
    retail = pd.read_csv("onlineretail.csv", encoding='ISO-8859-1')
    
    # Drop rows where Quantity or UnitPrice has a negative value
    retail = retail[(retail['Quantity'] >= 0) & (retail['UnitPrice'] >= 0)]

    # Remove records with UnitPrice <= 0
    retail = retail[retail['UnitPrice'] > 0]

    # Handling missing values
    retail.dropna(inplace=True)
    
    return retail

retail = load_data()

# Preprocess data
retail['TotalAmount'] = retail['Quantity'] * retail['UnitPrice']
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d/%m/%Y')

# Group by CustomerID and calculate Recency, Frequency, Monetary
today = datetime(2011, 12, 9)
df_recency = retail.groupby('CustomerID').agg({'InvoiceDate': lambda x: (today - x.max()).days}).reset_index()
df_recency.columns = ['CustomerID', 'Recency']

df_frequency = retail.groupby(by='CustomerID', as_index=False)['InvoiceNo'].count()
df_frequency.columns = ['CustomerID', 'Frequency']

df_monetary = retail.groupby(by='CustomerID',as_index=False).agg({'TotalAmount': 'sum'})
df_monetary.columns = ['CustomerID', 'Monetary']

rfm = pd.merge(pd.merge(df_recency, df_frequency, on='CustomerID'), df_monetary, on='CustomerID')

# Define RFM scoring functions
quantiles = rfm.quantile(q=[0.25, 0.5, 0.75]).to_dict()

def Rscore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

def FMscore(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1

rfm['R'] = rfm['Recency'].apply(Rscore, args=('Recency', quantiles))
rfm['F'] = rfm['Frequency'].apply(FMscore, args=('Frequency', quantiles))
rfm['M'] = rfm['Monetary'].apply(FMscore, args=('Monetary', quantiles))

# Calculate RFM Score
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)

# Define customer levels
bins = [0, 4, 7, 10, 12]
labels = ['Diamond', 'Emerald', 'Sapphire', 'Ruby']
rfm['Level'] = pd.cut(rfm['RFM_Score'], bins=bins, labels=labels, right=True)

# Apply log transformation for clustering using the handle_neg_n_zero function
def handle_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num

# Apply the handle_neg_n_zero function to Recency, Frequency, and Monetary columns
rfm['Recency'] = rfm['Recency'].apply(handle_neg_n_zero)
rfm['Monetary'] = rfm['Monetary'].apply(handle_neg_n_zero)
rfm['Frequency'] = rfm['Frequency'].apply(handle_neg_n_zero)

# Apply log transformations to the handled data
rfm['Recency_Log'] = rfm['Recency'].apply(math.log)
rfm['Frequency_Log'] = rfm['Frequency'].apply(math.log)
rfm['Monetary_Log'] = rfm['Monetary'].apply(math.log)

st.markdown("### Most recent purchase date")
# Recency input via slider for date range
recency_slider = st.slider(
    "Select your most recent purchase date",
    min_value=today - timedelta(days=365),
    max_value=today,
    value=today - timedelta(days=30),
    format="DD/MM/YYYY"
)
recency_days = (today - recency_slider).days
st.write(f"Recency: {recency_days} days")

# Frequency input with slider
st.markdown("### Number of Visits to the Store")
frequency_input = st.slider("Select how many times you visited the store", min_value=1, max_value=380, value=10)
st.write(f"Frequency: {frequency_input}")

# Monetary input with slider
st.markdown("### Total Amount Spent")
monetary_input = st.slider("Select the approximate amount of money spent", min_value=1.0, max_value=300000.0, value=5.0, step=10.0)
st.write(f"Monetary: {monetary_input}")

# Assign R, F, M based on user input using the quantiles
user_r = Rscore(recency_days, 'Recency', quantiles)
user_f = FMscore(frequency_input, 'Frequency', quantiles)
user_m = FMscore(monetary_input, 'Monetary', quantiles)

st.markdown("### Customer's Loyalty Level")
rankdata = {
    'RFM Score' : ['3 - 4', '5 - 7' , '8 - 10', '11 - 12'],
    'Level': ['Diamond', 'Emerald', 'Sapphire', 'Ruby']
}

df = pd.DataFrame(rankdata)
st.table(df)

rfm_score = user_r + user_f + user_m
st.write(f"Your RFM Score: {rfm_score}")
# Determine customer level
level = pd.cut([rfm_score], bins=bins, labels=labels)[0]
st.write(f"Customer Level: {level}")

# Function to handle negative and zero values before applying log
def handle_neg_n_zero(num):
    if num <= 0:
        return 1  # Return 1 to avoid log(0) or log of negative values
    else:
        return num

# Apply the handle_neg_n_zero function to user inputs
recency_input_handled = handle_neg_n_zero(recency_days)
frequency_input_handled = handle_neg_n_zero(frequency_input)
monetary_input_handled = handle_neg_n_zero(monetary_input)

# Apply log transformation to the handled user inputs
recency_log_input = math.log(recency_input_handled)
frequency_log_input = math.log(frequency_input_handled)
monetary_log_input = math.log(monetary_input_handled)


# Clustering on Recency, Frequency, and Monetary (3 features)
X = rfm[['Recency_Log', 'Frequency_Log', 'Monetary_Log']]
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Clustering on Recency and Monetary (2 features)
Y = rfm[['Recency_Log', 'Monetary_Log']]
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# Clustering on Frequency and Monetary (2 features)
Z = rfm[['Frequency_Log', 'Monetary_Log']]
scaler_Z = StandardScaler()
Z_scaled = scaler_Z.fit_transform(Z)

st.markdown("### Number of Cluster")
# Allow user to select number of clusters but limit it
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=min(10, len(rfm)), value=3)

# Clustering with Recency, Frequency, and Monetary (X)
if len(X) >= num_clusters:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_scaled)
    rfm['Cluster'] = kmeans.labels_

    # Plot clusters (Recency vs Frequency)
    st.markdown("### Recency vs Frequency")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency_Log', y='Frequency_Log', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Customer Segmentation based on Recency and Frequency")
    st.pyplot(fig)

     # Silhouette Score
    if len(np.unique(kmeans.labels_)) > 1:
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        st.write(f"Silhouette Score: {sil_score:.2f}")
    else:
        st.write("Not enough distinct clusters for silhouette score.")

    # Transform and predict for user input (3 features)
    user_log_data = scaler_X.transform([[recency_log_input, frequency_log_input, monetary_log_input]])
    user_cluster = kmeans.predict(user_log_data)
    st.write(f"Based on the input, you belong to Cluster: {user_cluster[0]}")

# Clustering with Recency and Monetary (Y)
if len(Y) >= num_clusters:
    kmeans2 = KMeans(n_clusters=num_clusters)
    kmeans2.fit(Y_scaled)
    rfm['Cluster'] = kmeans2.labels_

    # Plot clusters (Recency vs Monetary)
    st.markdown("### Recency vs Monetary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency_Log', y='Monetary_Log', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Customer Segmentation based on Recency and Monetary")
    st.pyplot(fig)

    # Silhouette Score
    if len(np.unique(kmeans2.labels_)) > 1:
        sil_score = silhouette_score(Y_scaled, kmeans2.labels_)
        st.write(f"Silhouette Score: {sil_score:.2f}")
    else:
        st.write("Not enough distinct clusters for silhouette score.")

    # Transform and predict for user input (2 features)
    user_log_data_Y = scaler_Y.transform([[recency_log_input, monetary_log_input]])
    user_cluster_Y = kmeans2.predict(user_log_data_Y)
    st.write(f"Based on the input, you belong to Cluster: {user_cluster_Y[0]}")

# Clustering with Frequency and Monetary (Z)
if len(Z) >= num_clusters:
    kmeans3 = KMeans(n_clusters=num_clusters)
    kmeans3.fit(Z_scaled)
    rfm['Cluster'] = kmeans3.labels_

    # Plot clusters (Frequency vs Monetary)
    st.markdown("### Frequency vs Monetary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Frequency_Log', y='Monetary_Log', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Customer Segmentation based on Frequency and Monetary")
    st.pyplot(fig)

    # Silhouette Score
    if len(np.unique(kmeans3.labels_)) > 1:
        sil_score = silhouette_score(Z_scaled, kmeans3.labels_)
        st.write(f"Silhouette Score: {sil_score:.2f}")
    else:
        st.write("Not enough distinct clusters for silhouette score.")

    # Transform and predict for user input (2 features)
    user_log_data_Z = scaler_Z.transform([[frequency_log_input, monetary_log_input]])
    user_cluster_Z = kmeans3.predict(user_log_data_Z)
    st.write(f"Based on the input, you belong to Cluster: {user_cluster_Z[0]}")
else:
    st.write("Not enough data points for clustering.")
