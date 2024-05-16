#Import necessary libraries
import plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

#create dataframe from csv 
breast_cancer_dataframe = pd.read_csv('data.csv')

#get features information
#breast_cancer_dataframe.info()

#extract a few sample data to have a look
breast_cancer_dataframe.head()

#data cleaning step - remove the columns or rows with missing values and the ID as it doesn't have any relevance in anaysis
breast_cancer_df = breast_cancer_dataframe.drop(['id','Unnamed: 32'], axis = 1)

#dropping the column called diagnosis and having a columns of 0 and 1 instead --> 1 for M(Malignant) and 0 for B(Benign)
breast_cancer_df= pd.get_dummies(breast_cancer_df,'diagnosis',drop_first=True) 

#check if new column is added and contains 0 and 1
breast_cancer_df.head()

#First Plotly chart - Bar chart to see the count of Malignant and Benign in our data

#create data to feed into the plot - x-axis will hold the name of diagnosis
#and y axis will have the counts according the number of matches found in diagnosis column in our dataframe

color = ['red','green']
data = [go.Bar(x=['Malignant','Benign'],
y=[breast_cancer_dataframe.loc[breast_cancer_dataframe['diagnosis']=='M'].shape[0],
   breast_cancer_dataframe.loc[breast_cancer_dataframe['diagnosis']=='B'].shape[0]],
   marker=dict(color=color) 

)]

#create the layout of the chart by defining titles for chart, x-axis and y-axis
layout = go.Layout(title='Breast Cancer - Diagnosis',
xaxis=dict(title='Diagnosis'),
yaxis=dict(title='Number of people')
)

#Imbed data and layout into charts figure using Figure function
fig = go.Figure(data=data, layout=layout)

#Use plot function of plotly to visualize the data
py.offline.plot(fig)

#breast_cancer_df.std()

'''
data = [go.Bar(x=breast_cancer_dataframe['radius_mean'],
y= breast_cancer_dataframe['texture_mean'])]

layout = go.Layout(title='Radius Mean v/s Texture Mean',
xaxis=dict(title='Radius Mean '),
yaxis=dict(title='Texture Mean')
)

fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig)
'''

#Heatmap - to visualize the correlation between features/factors given in the dataset

#calculate the pairwise correlation of columns - Pearson correlation coefficient. 
z = breast_cancer_df.corr()

#use Heatmap function available in plotly and create trace(collection of data) for plot
trace = go.Heatmap(
            x=z.index,       #set x as the feature id/name
            y=z.index,       #set y as the feature id/name
            z=z.values,      #set z as the correlation matrix values, 
                             #these values will be used to show the coloring on heatmap,
                             #which will eventually define which coefficient has more impact or are closly related
            colorscale='Viridis', #colorscale to define different colors for different range of values in correlation matrix
    )

#set the title of the plot
title = "plotting the correlation matrix of the breast cancer dataset"

##create the layout of the chart by defining title for chart, height and width of it
layout = go.Layout(
    title=title,          # set plot title
    autosize=False,       # turn off autosize 
    height=800,           # plot's height in pixels 
    width=800             # plot's height in pixels 
)

#covert the trace into list object before passing thru Figure function
data = go.Data([trace])

#Imbed data and layout into plot using Figure function
fig = go.Figure(data=data, layout=layout)

#Use plot function of plotly to visualize the data
py.offline.plot(fig)

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

sns.countplot(x='diagnosis',data = breast_cancer_dataframe,palette='BrBG')

plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(breast_cancer_df.corr())



