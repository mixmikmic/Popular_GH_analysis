# import the necessary packages
import pandas as pd
import numpy as np

import geoplotlib
from geoplotlib.colors import ColorMap
from geoplotlib.colors import create_set_cmap
import pyglet
from sklearn.cluster import KMeans
from geoplotlib.layers import BaseLayer
from geoplotlib.core import BatchPainter
from geoplotlib.utils import BoundingBox

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

df = pd.read_csv('2015.csv')
print(df.head())
map_data = pd.read_csv('countries.csv')
map_data.head()

scl = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(240, 210, 250)"]]


# scl = [[0.0, 'rgb(50,10,143)'],[0.2, 'rgb(117,107,177)'],[0.4, 'rgb(158,154,200)'],\
#             [0.6, 'rgb(188,189,220)'],[0.8, 'rgb(218,208,235)'],[1.0, 'rgb(250,240,255)']]
data = dict(type = 'choropleth', 
            colorscale = scl,
            autocolorscale = True,
            reversescale = True,
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Happiness Rank'], 
           text = df['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Orthographic'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)

df = df.merge(map_data, how='left', on = ['Country'])
df.head()

df['Happiness Score'].min(), df['Happiness Score'].max()

df['text']=df['Country'] + '<br>Happiness Score ' + (df['Happiness Score']).astype(str)
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'country names',
        lon = df['Longitude'],
        lat = df['Latitude'],
        text = df['text'],
        mode = 'markers',
        marker = dict(
            size = df['Happiness Score']*3,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df['Economy (GDP per Capita)'],
            cmax = df['Economy (GDP per Capita)'].max(),
            colorbar=dict(
                title="GDP per Capita"
            )
        ))]

layout = dict(
        title = 'Happiness Scores by GDP',
        geo = dict(
#             scope='usa',
            projection=dict( type='Mercator' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
#             subunitcolor = "rgb(217, 217, 217)",
#             countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

symbolmap = go.Figure(data = data, layout=layout)

iplot(symbolmap)

"""
Example of keyboard interaction
"""

class KMeansLayer(BaseLayer):

    def __init__(self, data):
        self.data = data
        self.k = 2


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['Longitude'], self.data['Latitude'])

        k_means = KMeans(n_clusters=self.k)
        k_means.fit(np.vstack([x,y]).T)
        labels = k_means.labels_

        self.cmap = create_set_cmap(set(labels), 'hsv')
        for l in set(labels):
            self.painter.set_color(self.cmap[l])
            self.painter.convexhull(x[labels == l], y[labels == l])
            self.painter.points(x[labels == l], y[labels == l], 2)
    
            
    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        ui_manager.info('Use left and right to increase/decrease the number of clusters. k = %d' % self.k)
        self.painter.batch_draw()


    def on_key_release(self, key, modifiers):
        if key == pyglet.window.key.LEFT:
            self.k = max(2,self.k - 1)
            return True
        elif key == pyglet.window.key.RIGHT:
            self.k = self.k + 1
            return True
        return False
  



data = geoplotlib.utils.DataAccessObject(df)
geoplotlib.add_layer(KMeansLayer(data))
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(geoplotlib.utils.BoundingBox.DK)
geoplotlib.show()



