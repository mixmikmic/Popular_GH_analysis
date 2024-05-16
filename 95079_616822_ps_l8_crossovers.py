# Notebook dependencies
from __future__ import print_function

import datetime
import json
import os

import ipyleaflet as ipyl
import ipywidgets as ipyw
from IPython.core.display import HTML
from IPython.display import display
import pandas as pd
from planet import api
from planet.api import filters
from shapely import geometry as sgeom

aoi = {u'geometry': {u'type': u'Polygon', u'coordinates': [[[-121.3113248348236, 38.28911976564886], [-121.3113248348236, 38.34622533958], [-121.2344205379486, 38.34622533958], [-121.2344205379486, 38.28911976564886], [-121.3113248348236, 38.28911976564886]]]}, u'type': u'Feature', u'properties': {u'style': {u'opacity': 0.5, u'fillOpacity': 0.2, u'noClip': False, u'weight': 4, u'color': u'blue', u'lineCap': None, u'dashArray': None, u'smoothFactor': 1, u'stroke': True, u'fillColor': None, u'clickable': True, u'lineJoin': None, u'fill': True}}}

json.dumps(aoi)

# define the date range for imagery
start_date = datetime.datetime(year=2017,month=1,day=1)
stop_date = datetime.datetime(year=2017,month=8,day=23)

# filters.build_search_request() item types:
# Landsat 8 - 'Landsat8L1G'
# Sentinel - 'Sentinel2L1C'
# PS Orthotile = 'PSOrthoTile'

def build_landsat_request(aoi_geom, start_date, stop_date):
    query = filters.and_filter(
        filters.geom_filter(aoi_geom),
        filters.range_filter('cloud_cover', lt=5),
        # ensure has all assets, unfortunately also filters 'L1TP'
#         filters.string_filter('quality_category', 'standard'), 
        filters.range_filter('sun_elevation', gt=0), # filter out Landsat night scenes
        filters.date_range('acquired', gt=start_date),
        filters.date_range('acquired', lt=stop_date)
    )

    return filters.build_search_request(query, ['Landsat8L1G'])    
    
    
def build_ps_request(aoi_geom, start_date, stop_date):
    query = filters.and_filter(
        filters.geom_filter(aoi_geom),
        filters.range_filter('cloud_cover', lt=0.05),
        filters.date_range('acquired', gt=start_date),
        filters.date_range('acquired', lt=stop_date)
    )

    return filters.build_search_request(query, ['PSOrthoTile'])

print(build_landsat_request(aoi['geometry'], start_date, stop_date))
print(build_ps_request(aoi['geometry'], start_date, stop_date))

def get_api_key():
    return os.environ['PL_API_KEY']

# quick check that key is defined
assert get_api_key(), "PL_API_KEY not defined."

def create_client():
    return api.ClientV1(api_key=get_api_key())

def search_pl_api(request, limit=500):
    client = create_client()
    result = client.quick_search(request)
    
    # note that this returns a generator
    return result.items_iter(limit=limit)


items = list(search_pl_api(build_ps_request(aoi['geometry'], start_date, stop_date)))
print(len(items))
# uncomment below to see entire metadata for a PS orthotile
# print(json.dumps(items[0], indent=4))
del items

items = list(search_pl_api(build_landsat_request(aoi['geometry'], start_date, stop_date)))
print(len(items))
# uncomment below to see entire metadata for a landsat scene
# print(json.dumps(items[0], indent=4))
del items

def items_to_scenes(items):
    item_types = []

    def _get_props(item):
        props = item['properties']
        props.update({
            'thumbnail': item['_links']['thumbnail'],
            'item_type': item['properties']['item_type'],
            'id': item['id'],
            'acquired': item['properties']['acquired'],
            'footprint': item['geometry']
        })
        return props
    
    scenes = pd.DataFrame(data=[_get_props(i) for i in items])
    
    # acquired column to index, it is unique and will be used a lot for processing
    scenes.index = pd.to_datetime(scenes['acquired'])
    del scenes['acquired']
    scenes.sort_index(inplace=True)
    
    return scenes

scenes = items_to_scenes(search_pl_api(build_landsat_request(aoi['geometry'],
                                                             start_date, stop_date)))
# display(scenes[:1])
print(scenes.thumbnail.tolist()[0])
del scenes

landsat_scenes = items_to_scenes(search_pl_api(build_landsat_request(aoi['geometry'],
                                                                     start_date, stop_date)))

# How many Landsat 8 scenes match the query?
print(len(landsat_scenes))

def landsat_scenes_to_features_layer(scenes):
    features_style = {
            'color': 'grey',
            'weight': 1,
            'fillColor': 'grey',
            'fillOpacity': 0.15}

    features = [{"geometry": r.footprint,
                 "type": "Feature",
                 "properties": {"style": features_style,
                                "wrs_path": r.wrs_path,
                                "wrs_row": r.wrs_row}}
                for r in scenes.itertuples()]
    return features

def create_landsat_hover_handler(scenes, label):
    def hover_handler(event=None, id=None, properties=None):
        wrs_path = properties['wrs_path']
        wrs_row = properties['wrs_row']
        path_row_query = 'wrs_path=={} and wrs_row=={}'.format(wrs_path, wrs_row)
        count = len(scenes.query(path_row_query))
        label.value = 'path: {}, row: {}, count: {}'.format(wrs_path, wrs_row, count)
    return hover_handler


def create_landsat_feature_layer(scenes, label):
    
    features = landsat_scenes_to_features_layer(scenes)
    
    # Footprint feature layer
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    feature_layer = ipyl.GeoJSON(data=feature_collection)

    feature_layer.on_hover(create_landsat_hover_handler(scenes, label))
    return feature_layer

# Initialize map using parameters from above map
# and deleting map instance if it exists
try:
    del fp_map
except NameError:
    pass


zoom = 6
center = [38.28993659801203, -120.14648437499999] # lat/lon

# Create map, adding box drawing controls
# Reuse parameters if map already exists
try:
    center = fp_map.center
    zoom = fp_map.zoom
    print(zoom)
    print(center)
except NameError:
    pass

# Change tile layer to one that makes it easier to see crop features
# Layer selected using https://leaflet-extras.github.io/leaflet-providers/preview/
map_tiles = ipyl.TileLayer(url='http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png')
fp_map = ipyl.Map(
        center=center, 
        zoom=zoom,
        default_tiles = map_tiles
    )

label = ipyw.Label(layout=ipyw.Layout(width='100%'))
fp_map.add_layer(create_landsat_feature_layer(landsat_scenes, label)) # landsat layer
fp_map.add_layer(ipyl.GeoJSON(data=aoi)) # aoi layer
    
# Display map and label
ipyw.VBox([fp_map, label])

def time_diff_stats(group):
    time_diff = group.index.to_series().diff() # time difference between rows in group
    stats = {'median': time_diff.median(),
             'mean': time_diff.mean(),
             'std': time_diff.std(),
             'count': time_diff.count(),
             'min': time_diff.min(),
             'max': time_diff.max()}
    return pd.Series(stats)

landsat_scenes.groupby(['wrs_path', 'wrs_row']).apply(time_diff_stats)

def find_closest(date_time, data_frame):
    # inspired by:
    # https://stackoverflow.com/questions/36933725/pandas-time-series-join-by-closest-time
    time_deltas = (data_frame.index - date_time).to_series().reset_index(drop=True).abs()
    idx_min = time_deltas.idxmin()

    min_delta = time_deltas[idx_min]
    return (idx_min, min_delta)

def closest_time(group):
    '''group: data frame with acquisition time as index'''
    inquiry_date = datetime.datetime(year=2017,month=3,day=7)
    idx, _ = find_closest(inquiry_date, group)
    return group.index.to_series().iloc[idx]


# for accurate results, we look at the closest time for each path/row tile to a given time
# using just the first entry could result in a longer time gap between collects due to
# the timing of the first entries
landsat_scenes.groupby(['wrs_path', 'wrs_row']).apply(closest_time)

all_ps_scenes = items_to_scenes(search_pl_api(build_ps_request(aoi['geometry'], start_date, stop_date)))

# How many PS scenes match query?
print(len(all_ps_scenes))
all_ps_scenes[:1]

def aoi_overlap_percent(footprint, aoi):
    aoi_shape = sgeom.shape(aoi['geometry'])
    footprint_shape = sgeom.shape(footprint)
    overlap = aoi_shape.intersection(footprint_shape)
    return overlap.area / aoi_shape.area

overlap_percent = all_ps_scenes.footprint.apply(aoi_overlap_percent, args=(aoi,))
all_ps_scenes = all_ps_scenes.assign(overlap_percent = overlap_percent)
all_ps_scenes.head()

print(len(all_ps_scenes))
ps_scenes = all_ps_scenes[all_ps_scenes.overlap_percent > 0.20]
print(len(ps_scenes))

# Use PS acquisition year, month, and day as index and group by those indices
# https://stackoverflow.com/questions/14646336/pandas-grouping-intra-day-timeseries-by-date
daily_ps_scenes = ps_scenes.index.to_series().groupby([ps_scenes.index.year,
                                                       ps_scenes.index.month,
                                                       ps_scenes.index.day])

daily_count = daily_ps_scenes.agg('count')

# How many days is the count greater than 1?
daily_multiple_count = daily_count[daily_count > 1]

print('out of {} days of coverage, {} days have multiple collects.'.format(     len(daily_count), len(daily_multiple_count)))

daily_multiple_count

def scenes_and_count(group):
    entry = {'count': len(group),
             'acquisition_time': group.index.tolist()}
    
    return pd.DataFrame(entry)

daily_count_and_scenes = daily_ps_scenes.apply(scenes_and_count)

multiplecoverage = daily_count_and_scenes.query('count > 1')

# multiplecoverage  # look at all occurrence
multiplecoverage.query('ilevel_1 == 7')  # look at just occurrence in July

def find_crossovers(acquired_time, landsat_scenes):
    '''landsat_scenes: pandas dataframe with acquisition time as index'''
    closest_idx, closest_delta = find_closest(acquired_time, landsat_scenes)
    closest_landsat = landsat_scenes.iloc[closest_idx]

    crossover = {'landsat_acquisition': closest_landsat.name,
                 'delta': closest_delta}
    return pd.Series(crossover)


# fetch PS scenes
ps_scenes = items_to_scenes(search_pl_api(build_ps_request(aoi['geometry'],
                                                           start_date, stop_date)))


# for each PS scene, find the closest Landsat scene
crossovers = ps_scenes.index.to_series().apply(find_crossovers, args=(landsat_scenes,))

# filter to crossovers within 1hr
concurrent_crossovers = crossovers[crossovers['delta'] < pd.Timedelta('1 hours')]
print(len(concurrent_crossovers))
concurrent_crossovers

def get_crossover_info(crossovers, aoi):
    def get_scene_info(acquisition_time, scenes):
        scene = scenes.loc[acquisition_time]
        scene_info = {'id': scene.id,
                      'thumbnail': scene.thumbnail,
                      # we are going to use the footprints as shapes so convert to shapes now
                      'footprint': sgeom.shape(scene.footprint)}
        return pd.Series(scene_info)

    landsat_info = crossovers.landsat_acquisition.apply(get_scene_info, args=(landsat_scenes,))
    ps_info = crossovers.index.to_series().apply(get_scene_info, args=(ps_scenes,))

    footprint_info = pd.DataFrame({'landsat': landsat_info.footprint,
                                   'ps': ps_info.footprint})
    overlaps = footprint_info.apply(lambda x: x.landsat.intersection(x.ps),
                                    axis=1)
    
    aoi_shape = sgeom.shape(aoi['geometry'])
    overlap_percent = overlaps.apply(lambda x: x.intersection(aoi_shape).area / aoi_shape.area)
    crossover_info = pd.DataFrame({'overlap': overlaps,
                                   'overlap_percent': overlap_percent,
                                   'ps_id': ps_info.id,
                                   'ps_thumbnail': ps_info.thumbnail,
                                   'landsat_id': landsat_info.id,
                                   'landsat_thumbnail': landsat_info.thumbnail})
    return crossover_info

crossover_info = get_crossover_info(concurrent_crossovers, aoi)
print(len(crossover_info))

significant_crossovers_info = crossover_info[crossover_info.overlap_percent > 0.9]
print(len(significant_crossovers_info))
significant_crossovers_info

def group_by_day(data_frame):
    return data_frame.groupby([data_frame.index.year,
                               data_frame.index.month,
                               data_frame.index.day])

unique_crossover_days = group_by_day(significant_crossovers_info.index.to_series()).count()
print(len(unique_crossover_days))
print(unique_crossover_days)

# https://stackoverflow.com/questions/36006136/how-to-display-images-in-a-row-with-ipython-display
def make_html(image):
     return '<img src="{}" style="display:inline;margin:1px"/>'             .format(image)


def display_thumbnails(row):
    print(row.name)
    display(HTML(''.join(make_html(t)
                         for t in (row.ps_thumbnail, row.landsat_thumbnail))))

_ = significant_crossovers_info.apply(display_thumbnails, axis=1)



