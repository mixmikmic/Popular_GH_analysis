# Notebook dependencies
from __future__ import print_function

import datetime
import copy
from functools import partial
import os

from IPython.display import display, Image
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from planet import api
from planet.api import filters
import pyproj
import rasterio
from rasterio import features as rfeatures
from shapely import geometry as sgeom
import shapely.ops

get_ipython().magic('matplotlib inline')

aoi = {u'geometry': {u'type': u'Polygon', u'coordinates': [[[-121.3113248348236, 38.28911976564886], [-121.3113248348236, 38.34622533958], [-121.2344205379486, 38.34622533958], [-121.2344205379486, 38.28911976564886], [-121.3113248348236, 38.28911976564886]]]}, u'type': u'Feature', u'properties': {u'style': {u'opacity': 0.5, u'fillOpacity': 0.2, u'noClip': False, u'weight': 4, u'color': u'blue', u'lineCap': None, u'dashArray': None, u'smoothFactor': 1, u'stroke': True, u'fillColor': None, u'clickable': True, u'lineJoin': None, u'fill': True}}}

# this notebook uses rasterio Shapes for processing, so lets convert that geojson to a shape
aoi_shape = sgeom.shape(aoi['geometry'])

def build_request(aoi_shape):
    old = datetime.datetime(year=2016,month=6,day=1)
    new = datetime.datetime(year=2016,month=10,day=1)

    query = filters.and_filter(
        filters.geom_filter(sgeom.mapping(aoi_shape)),
        filters.range_filter('cloud_cover', lt=5),
        filters.date_range('acquired', gt=old),
        filters.date_range('acquired', lt=new)
    )
    
    item_types = ['PSOrthoTile']
    return filters.build_search_request(query, item_types)

request = build_request(aoi_shape)
print(request)

# Utility functions: projecting a feature to the appropriate UTM zone

def get_utm_projection_fcn(shape):
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), #wgs84
        _get_utm_projection(shape))
    return proj_fcn


def _get_utm_zone(shape):
    '''geom: geojson geometry'''
    centroid = shape.centroid
    lon = centroid.x
    lat = centroid.y
    
    if lat > 84 or lat < -80:
        raise Exception('UTM Zones only valid within [-80, 84] latitude')
    
    # this is adapted from
    # https://www.e-education.psu.edu/natureofgeoinfo/book/export/html/1696
    zone = int((lon + 180) / 6 + 1)
    
    hemisphere = 'north' if lat > 0 else 'south'
    
    return (zone, hemisphere)


def _get_utm_projection(shape):
    zone, hemisphere = _get_utm_zone(shape)
    proj_str = "+proj=utm +zone={zone}, +{hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(
        zone=zone, hemi=hemisphere)
    return pyproj.Proj(proj_str)


proj_fcn = get_utm_projection_fcn(aoi_shape)
aoi_shape_utm = shapely.ops.transform(proj_fcn, aoi_shape)
print(aoi_shape_utm)

def get_coverage_dimensions(aoi_shape_utm):
    '''Checks that aoi is big enough and calculates the dimensions for coverage grid.'''
    minx, miny, maxx, maxy = aoi_shape_utm.bounds
    width = maxx - minx
    height = maxy - miny
    
    min_cell_size = 9 # in meters, approx 3x ground sampling distance
    min_number_of_cells = 3
    max_number_of_cells = 3000
    
    
    min_dim = min_cell_size * min_number_of_cells
    if height < min_dim:
        raise Exception('AOI height too small, should be {}m.'.format(min_dim))

    if width < min_dim:
        raise Exception('AOI width too small, should be {}m.'.format(min_dim))
    
    def _dim(length):
        return min(int(length/min_cell_size), max_number_of_cells)

    return [_dim(l) for l in (height, width)]


dimensions = get_coverage_dimensions(aoi_shape_utm)
print(dimensions)

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

def get_overlap_shapes_utm(items, aoi_shape):
    '''Determine overlap between item footprint and AOI in UTM.'''
    
    proj_fcn = get_utm_projection_fcn(aoi_shape)
    aoi_shape_utm = shapely.ops.transform(proj_fcn, aoi_shape)

    def _calculate_overlap(item):
        footprint_shape = sgeom.shape(item['geometry'])
        footprint_shape_utm = shapely.ops.transform(proj_fcn, footprint_shape)
        return aoi_shape_utm.intersection(footprint_shape_utm)

    for i in items:
        yield _calculate_overlap(i)


items = search_pl_api(request)

# cache the overlaps as a list so we don't have to refetch items
overlaps = list(get_overlap_shapes_utm(items, aoi_shape))
print(len(overlaps))

# what do overlaps look like?
# lets just look at the first overlap to avoid a long output cell
display(overlaps[0])

def calculate_coverage(overlaps, dimensions, bounds):
    
    # get dimensions of coverage raster
    mminx, mminy, mmaxx, mmaxy = bounds

    y_count, x_count = dimensions
    
    # determine pixel width and height for transform
    width = (mmaxx - mminx) / x_count
    height = (mminy - mmaxy) / y_count # should be negative

    # Affine(a, b, c, d, e, f) where:
    # a = width of a pixel
    # b = row rotation (typically zero)
    # c = x-coordinate of the upper-left corner of the upper-left pixel
    # d = column rotation (typically zero)
    # e = height of a pixel (typically negative)
    # f = y-coordinate of the of the upper-left corner of the upper-left pixel
    # ref: http://www.perrygeo.com/python-affine-transforms.html
    transform = rasterio.Affine(width, 0, mminx, 0, height, mmaxy)
    
    coverage = np.zeros(dimensions, dtype=np.uint16)
    for overlap in overlaps:
        if not overlap.is_empty:
            # rasterize overlap vector, transforming to coverage raster
            # pixels inside overlap have a value of 1, others have a value of 0
            overlap_raster = rfeatures.rasterize(
                    [sgeom.mapping(overlap)],
                    fill=0,
                    default_value=1,
                    out_shape=dimensions,
                    transform=transform)
            
            # add overlap raster to coverage raster
            coverage += overlap_raster
    return coverage


# what is a low-resolution look at the coverage grid?
display(calculate_coverage(overlaps, (6,3), aoi_shape_utm.bounds))

def plot_coverage(coverage):
    fig, ax = plt.subplots()
    cax = ax.imshow(coverage, interpolation='nearest', cmap=cm.viridis)
    ax.set_title('Coverage\n(median: {})'.format(int(np.median(coverage))))
    ax.axis('off')
    
    ticks_min = coverage.min()
    ticks_max = coverage.max()
    cbar = fig.colorbar(cax,ticks=[ticks_min, ticks_max])


plot_coverage(calculate_coverage(overlaps, dimensions, aoi_shape_utm.bounds))

demo_aoi = aoi  # use the same aoi that was used before

demo_aoi_shape = sgeom.shape(demo_aoi['geometry'])

proj_fcn = get_utm_projection_fcn(demo_aoi_shape)
demo_aoi_shape_utm = shapely.ops.transform(proj_fcn, demo_aoi_shape)
demo_dimensions = get_coverage_dimensions(demo_aoi_shape_utm)                               

# Parameterize our search request by start/stop dates for this comparison
def build_request_by_dates(aoi_shape, old, new):
    query = filters.and_filter(
        filters.geom_filter(sgeom.mapping(aoi_shape)),
        filters.range_filter('cloud_cover', lt=5),
        filters.date_range('acquired', gt=old),
        filters.date_range('acquired', lt=new)
    )
    
    item_types = ['PSOrthoTile']
    return filters.build_search_request(query, item_types)  

request_2016 = build_request_by_dates(demo_aoi_shape,
                                      datetime.datetime(year=2016,month=6,day=1),
                                      datetime.datetime(year=2016,month=8,day=1))                                    
items = search_pl_api(request_2016)
overlaps = list(get_overlap_shapes_utm(items, demo_aoi_shape))
plot_coverage(calculate_coverage(overlaps, demo_dimensions, demo_aoi_shape_utm.bounds))

request_2017 = build_request_by_dates(demo_aoi_shape,
                                      datetime.datetime(year=2017,month=6,day=1),
                                      datetime.datetime(year=2017,month=8,day=1))
items = search_pl_api(request_2017)
overlaps = list(get_overlap_shapes_utm(items, demo_aoi_shape))
plot_coverage(calculate_coverage(overlaps, demo_dimensions, demo_aoi_shape_utm.bounds))



