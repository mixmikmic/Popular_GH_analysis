import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.__version__

get_ipython().magic('matplotlib inline')

sales=pd.read_csv("sample-salesv2.csv",parse_dates=['date'])
sales.head()

sales.describe()

sales['unit price'].describe()

sales.dtypes

customers = sales[['name','ext price','date']]
customers.head()

customer_group = customers.groupby('name')
customer_group.size()

sales_totals = customer_group.sum()
sales_totals.sort_values(by=['ext price']).head()

my_plot = sales_totals.plot(kind='bar')

my_plot = sales_totals.sort_values(by=['ext price'],ascending=False).plot(kind='bar',legend=None,title="Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales ($)")

customers = sales[['name','category','ext price','date']]
customers.head()

category_group=customers.groupby(['name','category']).sum()
category_group.head()

category_group.unstack().head()

my_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")

my_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer",figsize=(9, 7))
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")
my_plot.legend(["Total","Belts","Shirts","Shoes"], loc=9,ncol=4)

purchase_patterns = sales[['ext price','date']]
purchase_patterns.head()

purchase_plot = purchase_patterns['ext price'].hist(bins=20)
purchase_plot.set_title("Purchase Patterns")
purchase_plot.set_xlabel("Order Amount($)")
purchase_plot.set_ylabel("Number of orders")

purchase_patterns = sales[['ext price','date']]
purchase_patterns.head()

purchase_patterns = purchase_patterns.set_index('date')
purchase_patterns.head()

purchase_plot = purchase_patterns.resample('M').sum().plot(title="Total Sales by Month",legend=None)

fig = purchase_plot.get_figure()
fig.savefig("total-sales.png")



