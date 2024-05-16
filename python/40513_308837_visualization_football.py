#Load all libraries
import os,sys  
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
from datashader import reductions
from datashader.colors import colormap_select, Hot, inferno
from datashader.bokeh_ext import InteractiveImage
from bokeh.palettes import Greens3, Blues3, Blues4, Blues9, Greys9
from bokeh.plotting import figure, output_notebook
from bokeh.tile_providers import WMTSTileSource, STAMEN_TONER, STAMEN_TERRAIN
from functools import partial
import wget
import zipfile
import math
from difflib import SequenceMatcher

output_notebook()
#print(sys.path)
print(sys.version)

df_stadium = pd.read_csv("stadiums.csv", usecols=['Team','Stadium','Latitude','Longitude','Country'])
print("Number of rows: %d" % df_stadium.shape[0])
dd1 = df_stadium.take([0,99, 64, 121])
dd1

df_match = pd.read_csv('champions.csv', usecols=['Date','home','visitor','hcountry','vcountry'])
df_match = df_match.rename(columns = {'hcountry':'home_country', 'vcountry':'visitor_country'})
df_teams_champions = pd.concat([df_match['home'], df_match['visitor']])
teams_champions = set(df_teams_champions)
print("Number of teams that have participated in the Champions League: %d" % len(teams_champions))
print("Number of matches in the dataset: %d" % df_match.shape[0])
df_match.head()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_info_similar_team(team, df_stadium, threshold=0.6, verbose=False):
    max_rank = 0
    max_idx = -1
    stadium = "Unknown"
    latitude = np.NaN
    longitude = np.NaN
    for idx, val in enumerate(df_stadium['Team']):
        rank = similar(team, val)
        if rank > threshold:
            if(verbose): print("%s and %s(Idx=%d) are %f similar." % (team, val, idx, rank))
            if rank > max_rank:
                if(verbose): print("New maximum rank: %f" %rank)
                max_rank = rank
                max_idx = idx
                stadium = df_stadium['Stadium'].iloc[max_idx]
                latitude = df_stadium['Latitude'].iloc[max_idx]
                longitude = df_stadium['Longitude'].iloc[max_idx]
    return stadium, latitude, longitude
print(get_info_similar_team("Real Madrid FC", df_stadium, verbose=True))
print(get_info_similar_team("Atletico de Madrid FC", df_stadium, verbose=True))
print(get_info_similar_team("Inter Milan", df_stadium, verbose=True))
 

get_ipython().run_cell_magic('time', '', "df_match_stadium = df_match\nhome_stadium_index = df_match_stadium['home'].map(lambda x: get_info_similar_team(x, df_stadium))\nvisitor_stadium_index = df_match_stadium['visitor'].map(lambda x: get_info_similar_team(x, df_stadium))\ndf_home = pd.DataFrame(home_stadium_index.tolist(), columns=['home_stadium', 'home_latitude', 'home_longitude'])\ndf_visitor = pd.DataFrame(visitor_stadium_index.tolist(), columns=['visitor_stadium', 'visitor_latitude', 'visitor_longitude'])\ndf_match_stadium = pd.concat([df_match_stadium, df_home, df_visitor], axis=1, ignore_index=False)")

print("Number of missing values for home teams: %d out of %d" % (df_match_stadium['home_stadium'].value_counts()['Unknown'], df_match_stadium.shape[0]))
df1 = df_match_stadium['home_stadium'] == 'Unknown'
df2 = df_match_stadium['visitor_stadium'] == 'Unknown'
n_complete_matches = df_match_stadium.shape[0] - df_match_stadium[df1 | df2].shape[0]
print("Number of matches with complete data: %d out of %d" % (n_complete_matches, df_match_stadium.shape[0]))
df_match_stadium.head()

def aggregate_dataframe_coordinates(dataframe):
    df = pd.DataFrame(index=np.arange(0, n_complete_matches*3), columns=['Latitude','Longitude'])
    count = 0
    for ii in range(dataframe.shape[0]):
        if dataframe['home_stadium'].loc[ii]!= 'Unknown' and dataframe['visitor_stadium'].loc[ii]!= 'Unknown':
            df.loc[count] = [dataframe['home_latitude'].loc[ii], dataframe['home_longitude'].loc[ii]]
            df.loc[count+1] = [dataframe['visitor_latitude'].loc[ii], dataframe['visitor_longitude'].loc[ii]]
            df.loc[count+2] = [np.NaN, np.NaN]
            count += 3
    return df
df_agg = aggregate_dataframe_coordinates(df_match_stadium)
df_agg.head()

def to_web_mercator(yLat, xLon):
    # Check if coordinate out of range for Latitude/Longitude
    if (abs(xLon) > 180) and (abs(yLat) > 90):  
        return
 
    semimajorAxis = 6378137.0  # WGS84 spheriod semimajor axis
    east = xLon * 0.017453292519943295
    north = yLat * 0.017453292519943295
 
    northing = 3189068.5 * math.log((1.0 + math.sin(north)) / (1.0 - math.sin(north)))
    easting = semimajorAxis * east
 
    return [easting, northing]
df_agg_mercator = df_agg.apply(lambda row: to_web_mercator(row['Latitude'], row['Longitude']), axis=1)
df_agg_mercator.head()

plot_width  = 850
plot_height = 600
x_range = (-1.9e6, 5.9e6)
y_range = (3.7e6, 9.0e6)
def create_image(x_range=x_range, y_range=y_range, w=plot_width, h=plot_height, cmap=None):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.line(df_agg_mercator, 'Latitude', 'Longitude',  ds.count())
    #img = tf.shade(agg, cmap=reversed(Blues3), how='eq_hist')
    #img = tf.shade(agg, cmap=reversed(Greens3), how='eq_hist')    
    img = tf.shade(agg, cmap=cmap, how='eq_hist')
    return img

def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width, plot_height=plot_height,**plot_args):
    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, **plot_args)
    
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    
    return p

ArcGIS=WMTSTileSource(url='http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{Z}/{Y}/{X}.png')
p = base_plot()
p.add_tile(ArcGIS)
#InteractiveImage(p, create_image, cmap=inferno)

df_stadium_read = pd.read_csv('stadiums_wikidata.csv', usecols=['clubLabel','venueLabel','coordinates','countryLabel'])
df_stadium_read.tail()

df_temp = df_stadium_read['coordinates'].str.extract('(?P<Longitude>[-+]?[0-9]*\.?[0-9]+) (?P<Latitude>[-+]?[0-9]*\.?[0-9]+)', expand=True)
df_stadium_new = pd.concat([df_stadium_read['clubLabel'],df_stadium_read['venueLabel'], df_temp, df_stadium_read['countryLabel']], axis=1) 
df_stadium_new = df_stadium_new.rename(columns = {'clubLabel':'Team', 'venueLabel':'Stadium','countryLabel':'Country'})
print("Number of rows: %d" % df_stadium_new.shape[0])
unique_teams_stadium = list(set(df_stadium_new['Team']))
print("Unique team's name number: %d" % len(unique_teams_stadium))
df_stadium_new.take(list(range(3388,3393)))

df_match_home = df_match[['home','home_country']]
df_match_home = df_match_home.rename(columns={'home':'Team','home_country':'Country'})
df_match_visitor = df_match[['visitor','visitor_country']]
df_match_visitor = df_match_visitor.rename(columns={'visitor':'Team','visitor_country':'Country'})
df_champions_teams = pd.concat([df_match_home,df_match_visitor], axis=0, ignore_index=True)
df_champions_teams = df_champions_teams.drop_duplicates()
print("Number of unique teams: %d" % df_champions_teams.shape[0])
country_dict = {'ALB':'Albania',
                'AND':'Andorra',
                'ARM':'Armenia',
                'AUT':'Austria',
                'AZE':'Azerbaijan',
                'BEL':'Belgium',
                'BIH':'Bosnia and Herzegovina',
                'BLR':'Belarus',
                'BUL':'Bulgaria',
                'CRO':'Croatia',
                'CYP':'Cyprus',
                'CZE':'Czech Republic',
                'DEN':'Denmark',
                #'ENG':'England',
                'ENG':'United Kingdom',
                'ESP':'Spain',
                'EST':'Estonia',
                'FIN':'Finland',
                'FRA':'France',
                'FRO':'Feroe Islands',
                'GEO':'Georgia',
                'GER':'Germany',
                'GIB':'Gibraltar',
                'GRE':'Greece',
                'HUN':'Hungary',
                'ITA':'Italy',
                'IRL':'Ireland',
                'ISL':'Iceland',
                'ISR':'Israel',
                'KAZ':'Kazakhstan',
                'LTU':'Lithuania',
                'LUX':'Luxembourg',
                'LVA':'Latvia',
                'MDA':'Moldova',
                'MKD':'Macedonia',
                'MLT':'Malta',
                'MNE':'Montenegro',
                'NED':'Netherlands',
                #'NIR':'Northern Ireland',
                'NIR':'United Kingdom',
                'NOR':'Norwey',
                'POL':'Poland',
                'POR':'Portugal',
                'ROU':'Romania',
                'RUS':'Russia',
                #'SCO':'Scotland',
                'SCO':'United Kingdom',
                'SMR':'San Marino',
                'SRB':'Serbia',
                'SUI':'Switzerland',
                'SVK':'Slovakia',
                'SVN':'Slovenia',
                'SWE':'Sweden',
                'TUR':'Turkey',
                'UKR':'Ukrania',
                #'WAL':'Wales',
                'WAL':'United Kingdom'}
df_champions_teams['Country'].replace(country_dict, inplace=True)
#df_champions_teams.to_csv('match_unique.csv')# To check that the mapping is correct
df_champions_teams.sort_values(by='Team',inplace=True)
df_champions_teams = df_champions_teams.reset_index(drop=True)
df_champions_teams.head()

get_ipython().run_cell_magic('time', '', 'def get_info_similar_team_country(team, country, df_stadium, df, threshold, verbose):\n    team2 = "Unknown"\n    stadium = "Unknown"\n    latitude = np.NaN\n    longitude = np.NaN\n    cols = list(df)\n    for idx, val in enumerate(df_stadium[\'Team\']):\n        rank = similar(team, val)\n        if rank > threshold and country == df_stadium[\'Country\'].iloc[idx]:\n            if(verbose): print("%s and %s(Idx=%d) are %f similar and from the same country %s." \n                               % (team, val, idx, rank, country))\n            team2 = df_stadium[\'Team\'].iloc[idx]\n            stadium = df_stadium[\'Stadium\'].iloc[idx]\n            latitude = df_stadium[\'Latitude\'].iloc[idx]\n            longitude = df_stadium[\'Longitude\'].iloc[idx]\n            dtemp = pd.DataFrame([[team, team2, stadium, latitude, longitude, country]], columns=cols)\n            df = df.append(dtemp, ignore_index=True)\n    #if there is no match, register it\n    if(team2 == "Unknown"):\n        df_nomatch = pd.DataFrame([[team, team2, stadium, latitude, longitude, country]], columns=cols)\n        df = df.append(df_nomatch, ignore_index=True)\n    return df\n\ndef generate_new_stadium_dataset(df_champions_teams, df_stadium_new, threshold=0.6, verbose=False):\n    df = pd.DataFrame(columns=[\'Team\', \'Team2\', \'Stadium\', \'Latitude\',\'Longitude\',\'Country\'])\n    for idx, row in df_champions_teams.iterrows():\n        df = get_info_similar_team_country(row[\'Team\'],row[\'Country\'], df_stadium_new, df, \n                                           threshold=threshold, verbose=verbose)\n    return df\n\nverbose = False # You can change this to True to see all the combinations\nthreshold = 0.5\ndf_stadiums_champions = generate_new_stadium_dataset(df_champions_teams, df_stadium_new, threshold, verbose)\ndf_stadiums_champions.to_csv(\'stadiums_champions.csv\', index=False)')

df_stadiums_champions = pd.read_csv('stadiums_champions_filtered.csv', usecols=['Team','Stadium','Latitude','Longitude','Country'])
df_stadiums_champions.head()

df_match_stadium_new= df_match
home_stadium_index = df_match_stadium_new['home'].map(lambda x: get_info_similar_team(x, df_stadiums_champions))
visitor_stadium_index = df_match_stadium_new['visitor'].map(lambda x: get_info_similar_team(x, df_stadiums_champions))
df_home = pd.DataFrame(home_stadium_index.tolist(), columns=['home_stadium', 'home_latitude', 'home_longitude'])
df_visitor = pd.DataFrame(visitor_stadium_index.tolist(), columns=['visitor_stadium', 'visitor_latitude', 'visitor_longitude'])
df_match_stadium_new = pd.concat([df_match_stadium_new, df_home, df_visitor], axis=1, ignore_index=False)
df1 = df_match_stadium_new['home_stadium'] == 'Unknown'
df2 = df_match_stadium_new['visitor_stadium'] == 'Unknown'
n_complete_matches = df_match_stadium_new.shape[0] - df_match_stadium_new[df1 | df2].shape[0]
print("Number of matches with complete data: %d out of %d" % (n_complete_matches, df_match_stadium_new.shape[0]))
df_match_stadium_new.head()

df_agg = aggregate_dataframe_coordinates(df_match_stadium_new)
df_agg_mercator = df_agg.apply(lambda row: to_web_mercator(row['Latitude'], row['Longitude']), axis=1)
print("Number of rows: %d" % df_agg_mercator.shape[0])
df_agg_mercator.head()

get_ipython().run_cell_magic('time', '', '#InteractiveImage(p, create_image, cmap=inferno)')

ArcGIS2 = WMTSTileSource(url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png')
p2 = base_plot()
p2.add_tile(ArcGIS2)
cmap = reversed(Blues3)
#cmap = reversed(Greens3)
#InteractiveImage(p2, create_image, cmap=cmap)

