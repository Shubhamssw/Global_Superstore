#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv("D:\Superstore\Global_Superstore2.csv",encoding = 'ISO-8859-1')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


import numpy as np


# In[7]:


category_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index()
category_sales_sorted = category_sales.sort_values(by='Sales', ascending=False)


# In[8]:


ascending=Falsetop_10_categories = category_sales_sorted.head(10)

print(Falsetop_10_categories)    


# In[9]:


def format_number_with_commas(number):
    return '{:,.2f}'.format(number)

# Create the bar graph
plt.figure(figsize=(10, 5))
sns.barplot(data=Falsetop_10_categories, x='Sub-Category', y='Sales', color='green')
plt.title('Top 10 Categories with Highest Sales')
plt.xticks(rotation=45)
plt.xlabel('Sub-Category')
plt.ylabel('Sales')

# Get the current y-axis ticks
y_ticks = plt.gca().get_yticks()

# Format y-axis ticks as actual numbers with commas
formatted_y_ticks = [format_number_with_commas(y) for y in y_ticks]

# Set the formatted y-axis tick labels
plt.gca().set_yticklabels(formatted_y_ticks)

plt.show()


# In[10]:


customer_sales = df.groupby('Customer Name')['Sales'].sum().reset_index()

customer_sales_sorted = customer_sales.sort_values(by='Sales', ascending=False)

top_10_customers = customer_sales_sorted.head(10)

print(top_10_customers) 


# In[11]:


plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_customers, x='Customer Name', y='Sales', color='orange')
plt.title('Top 10 Customers with Highest Sales')
plt.xticks(rotation=45)
plt.xlabel('Customer Name')
plt.ylabel('Sales')
plt.show()


# In[12]:


item_sales = df.groupby('Product Name')['Sales'].sum().reset_index()

item_sales_sorted = item_sales.sort_values(by='Sales', ascending=False)

top_5_items = item_sales_sorted.head(5)

print(top_5_items)


# In[13]:


plt.figure(figsize=(8, 4))
sns.barplot(data=top_5_items, x='Product Name', y='Sales', color='cyan')
plt.title('Top 5 Items with Highest Sales')
plt.xticks(rotation=45)
plt.xlabel('Product Name')
plt.ylabel('Sales')
plt.show()


# In[14]:


df['Profit Margin'] = ((df['Profit']+df['Shipping Cost']) / df['Sales']) * 100

item_profit_margin = df.groupby('Product Name')['Profit Margin'].mean().reset_index()

item_profit_margin_sorted = item_profit_margin.sort_values(by='Profit Margin')

top_5_low_profit_items = item_profit_margin_sorted.head(5)

print(top_5_low_profit_items)


# In[15]:


plt.figure(figsize=(12, 6))
sns.barplot(data=top_5_low_profit_items, x='Product Name', y='Profit Margin', color='purple')
plt.title('Top 5 Items with Lowest Profit Margin')
plt.xticks(rotation=45)
plt.xlabel('Product Name')
plt.ylabel('Profit Margin')
plt.show()


# In[16]:


item_profit_margin = df.groupby('Product Name')['Profit Margin'].sum().reset_index()

item_profit_margin_sorted = item_profit_margin.sort_values(by='Profit Margin', ascending=False)

top_5_high_profit_items = item_profit_margin_sorted.head(5)

print(top_5_high_profit_items)


# In[17]:


plt.figure(figsize=(12, 6))
sns.barplot(data=top_5_high_profit_items, x='Product Name', y='Profit Margin', color='black')
plt.title('Top 5 Items with Highest Profit Margin')
plt.xticks(rotation=45)
plt.xlabel('Product Name')
plt.ylabel('Profit Margin')
plt.show()


# In[18]:


df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Year'] = df['Order Date'].dt.year

yearly_sales = df.groupby('Year')['Sales'].sum()

print(yearly_sales)


# In[19]:


yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
fig = px.line(yearly_sales, x='Year', y='Sales', title='Yearly Sales')
fig.update_xaxes(type='category') 
yearly_sales = df.groupby('Year')['Sales']
fig.show()


# In[20]:


df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Year'] = df['Order Date'].dt.year
df['Pr'] = (df['Profit'] / df['Sales']) * 100
yearly_profit = df.groupby('Year')['Pr'].sum()

print(yearly_profit)


# In[21]:


yearly_profit = df.groupby(df['Order Date'].dt.year)['Profit'].sum().reset_index()
fig = px.line(yearly_profit, x='Order Date', y='Profit', title='Profit on Yearly Basis')
fig.update_xaxes(type='category')
fig.show()


# In[22]:


floats = []
for col in df.columns:
  if df[col].dtype == 'float':
    floats.append(col)
 
len(floats)


# In[23]:


plt.subplots(figsize = (15, 5))
for i, col in enumerate(floats):
    plt.subplot(2, 5, i + 1)
    sns.distplot(df[col])
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize = (10, 5))
sns.countplot(df['Profit Margin'])
plt.axis('off')
plt.show()


# In[25]:


df.shape


# In[26]:


df.info()


# In[27]:


df.isnull().sum()


# In[28]:


df.dropna(inplace = True)
df.isnull().sum().plot.bar()
plt.show()


# In[ ]:




