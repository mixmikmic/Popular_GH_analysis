import pandas as pd
import numpy as np

sales = pd.read_excel("https://github.com/chris1610/pbpython/blob/master/data/sales-estimate.xlsx?raw=True", sheetname="projections")
sales

print(sales["Current_Price"].mean())
print(sales["New_Product_Price"].mean())

print((sales["Current_Price"] * sales["Quantity"]).sum() / sales["Quantity"].sum())
print((sales["New_Product_Price"] * sales["Quantity"]).sum() / sales["Quantity"].sum())

print(np.average(sales["Current_Price"], weights=sales["Quantity"]))
print(np.average(sales["New_Product_Price"], weights=sales["Quantity"]))

def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

print(wavg(sales, "Current_Price", "Quantity"))
print(wavg(sales, "New_Product_Price", "Quantity"))

sales.groupby("Manager").apply(wavg, "Current_Price", "Quantity")

sales.groupby("Manager").apply(wavg, "New_Product_Price", "Quantity")

sales.groupby("State").apply(wavg, "New_Product_Price", "Quantity")

sales.groupby(["Manager", "State"]).apply(wavg, "New_Product_Price", "Quantity")

f = {'New_Product_Price': ['mean'],'Current_Price': ['median'], 'Quantity': ['sum', 'mean']}
sales.groupby("Manager").agg(f)

data_1 = sales.groupby("Manager").apply(wavg, "New_Product_Price", "Quantity")
data_2 = sales.groupby("Manager").apply(wavg, "Current_Price", "Quantity")

summary = pd.DataFrame(data=dict(s1=data_1, s2=data_2))
summary.columns = ["New Product Price","Current Product Price"]
summary.head()

np.average(sales["Current_Price"], weights=sales["Quantity"])

sales.groupby("Manager").apply(lambda x: np.average(x['New_Product_Price'], weights=x['Quantity']))



