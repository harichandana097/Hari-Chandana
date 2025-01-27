#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File URLs
file_urls = {
    "Customers.csv": "https://drive.google.com/uc?id=1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE",
    "Products.csv": "https://drive.google.com/uc?id=1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0",
    "Transactions.csv": "https://drive.google.com/uc?id=1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF",
}

# Read data into DataFrames
customers = pd.read_csv(file_urls["Customers.csv"])
products = pd.read_csv(file_urls["Products.csv"])
transactions = pd.read_csv(file_urls["Transactions.csv"])

customers
products
transactions


# In[ ]:





# In[ ]:





# In[4]:


# EDA: Initial Overview
print("=== Customers Data ===")
print(customers.head())
print(customers.info(), "\n")

print("=== Products Data ===")
print(products.head())
print(products.info(), "\n")

print("=== Transactions Data ===")
print(transactions.head())
print(transactions.info(), "\n")


# In[17]:


# Check for missing values in each column
print(transactions.isnull().sum())


# In[15]:


#merge
transactions = transactions.merge(products, on='ProductID', how='left', suffixes=('', '_product'))
transactions = transactions.merge(customers, on='CustomerID', how='left', suffixes=('', '_customer'))
print(transactions.head())


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of TotalValue (Revenue)
plt.figure(figsize=(10, 6))
sns.histplot(transactions['TotalValue'], kde=True, color='blue')
plt.title('Distribution of Total Revenue')
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.show()





# In[20]:


# Group by Category and sum the revenue (TotalValue)
category_revenue = transactions.groupby('Category_x')['TotalValue'].sum().sort_values(ascending=False)

# Plot top categories by revenue
plt.figure(figsize=(12, 6))
category_revenue.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Product Categories by Revenue')
plt.xlabel('Category')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[21]:


# Plot the relationship between TotalValue and Quantity sold
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='TotalValue', data=transactions, color='purple')
plt.title('Total Revenue vs Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Total Revenue')
plt.show()


# In[24]:


# Convert TransactionDate to datetime format
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Group by date and sum the revenue (TotalValue)
daily_revenue = transactions.groupby(transactions['TransactionDate'].dt.date)['TotalValue'].sum()

# Plot revenue trend over time
plt.figure(figsize=(12, 6))
daily_revenue.plot(kind='line', color='blue')
plt.title('Revenue Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.show()



# In[25]:


# Group by ProductName and sum the revenue (TotalValue)
product_revenue = transactions.groupby('ProductName_x')['TotalValue'].sum().sort_values(ascending=False)

# Plot top products by revenue
plt.figure(figsize=(12, 6))
product_revenue.head(10).plot(kind='bar', color='red')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Product')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[ ]:





# In[ ]:




