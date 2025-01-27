#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the Data

import pandas as pd

# URLs for the datasets
customers_url = "https://drive.google.com/uc?id=1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE"
products_url = "https://drive.google.com/uc?id=1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0"
transactions_url = "https://drive.google.com/uc?id=1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF"

# Load datasets
customers = pd.read_csv(customers_url)
products = pd.read_csv(products_url)
transactions = pd.read_csv(transactions_url)

# Display the first few rows of each dataset
print("Customers.csv:")
print(customers.head())

print("\nProducts.csv:")
print(products.head())

print("\nTransactions.csv:")
print(transactions.head())



# In[5]:


# Step 2: Merge Data

data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
data


# In[7]:


# Step 3: Feature Engineering
# Check the columns of the merged dataset
print(data.columns)

# Perform aggregation
customer_features = data.groupby("CustomerID").agg({
    "TotalValue": "sum",                  # Total spending
    "Quantity": "sum",                   # Total quantity purchased
    "Category": lambda x: x.mode()[0],   # Most frequent product category
    "Price_x": "mean"                    # Average price of products purchased (check 'Price_x' or 'Price_y')
}).reset_index()


# In[9]:


# Step 4: Compute Similarity
feature_matrix = customer_features.set_index("CustomerID").drop(columns=["Category"])
similarity_matrix = cosine_similarity(feature_matrix)
similarity_matrix


# In[11]:


# Step 5: Find Top 3 Lookalikes
customer_ids = customer_features["CustomerID"].tolist()
lookalikes = {}

for i, cust_id in enumerate(customer_ids[:20]):  # For the first 20 customers
    similarity_scores = list(enumerate(similarity_matrix[i]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 (excluding self)
    lookalikes[cust_id] = [(customer_ids[idx], round(score, 3)) for idx, score in similarity_scores]
    
lookalikes
    


# In[12]:


# Step 6: Save Lookalike Results
lookalike_data = [{"cust_id": key, "lookalikes": value} for key, value in lookalikes.items()]
lookalike_df = pd.DataFrame(lookalike_data)
lookalike_df.to_csv("Lookalike.csv", index=False)



# In[ ]:




