# Notebook dependencies
from __future__ import print_function

from collections import namedtuple
import copy
from functools import partial
import json
import os
from xml.dom import minidom

import cv2
import ipyleaflet as ipyl
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from shapely.geometry import mapping, shape, Polygon
from shapely.ops import transform
from sklearn import neighbors
from sklearn.metrics import classification_report

get_ipython().magic('matplotlib inline')

# Train data
train_data_folder = os.path.join('data', 'test')

# Created in datasets-identify notebook
train_pl_metadata_filename = os.path.join(train_data_folder, '20160831_180257_0e26_3B_AnalyticMS_metadata.xml')
assert os.path.isfile(train_pl_metadata_filename)

# Created in datasets-prepare notebook
train_pl_filename = os.path.join(train_data_folder, '20160831_180257_0e26_3B_AnalyticMS_cropped.tif')
assert os.path.isfile(train_pl_filename)

train_ground_truth_filename = os.path.join(train_data_folder, 'ground-truth_cropped.geojson')
assert os.path.isfile(train_ground_truth_filename)

# Test data
test_data_folder = os.path.join('data', 'train')

# Created in datasets-identify notebook
test_pl_metadata_filename = os.path.join(test_data_folder, '20160831_180231_0e0e_3B_AnalyticMS_metadata.xml')
assert os.path.isfile(test_pl_metadata_filename)

# Created in datasets-prepare notebook
test_pl_filename = os.path.join(test_data_folder, '20160831_180231_0e0e_3B_AnalyticMS_cropped.tif')
assert os.path.isfile(test_pl_filename)

test_ground_truth_filename = os.path.join(test_data_folder, 'ground-truth_cropped.geojson')
assert os.path.isfile(test_ground_truth_filename)

# Utility functions: features to contours

def project_feature(feature, proj_fcn):
    """Creates a projected copy of the feature.
    
    :param feature: geojson description of feature to be projected
    :param proj_fcn: partial function defining projection transform"""
    g1 = shape(feature['geometry'])
    g2 = transform(proj_fcn, g1)
    proj_feat = copy.deepcopy(feature)
    proj_feat['geometry'] = mapping(g2)
    return proj_feat


def project_features_to_srs(features, img_srs, src_srs='epsg:4326'):
    """Project features to img_srs.
    
    If src_srs is not specified, WGS84 (only geojson-supported crs) is assumed.
    
    :param features: list of geojson features to be projected
    :param str img_srs: destination spatial reference system
    :param str src_srs: source spatial reference system
    """
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init=src_srs),
        pyproj.Proj(init=img_srs))
    
    return [project_feature(f, proj_fcn) for f in features]


def polygon_to_contour(feature_geometry, image_transform):
    """Convert the exterior ring of a geojson Polygon feature to an
    OpenCV contour.
    
    image_transform is typically obtained from `img.transform` where 
    img is obtained from `rasterio.open()
    
    :param feature_geometry: the 'geometry' entry in a geojson feature
    :param rasterio.Affine image_transform: image transformation"""
    points_xy = shape(feature_geometry).exterior.coords
    points_x, points_y = zip(*points_xy) # inverse of zip
    rows, cols = rasterio.transform.rowcol(image_transform, points_x, points_y)
    return np.array([pnt for pnt in zip(cols, rows)], dtype=np.int32)

def load_geojson(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_contours(ground_truth_filename, pl_filename):
    with rasterio.open(pl_filename) as img:
        img_transform = img.transform
        img_srs = img.crs['init']

    ground_truth_data = load_geojson(ground_truth_filename)

    # project to image srs
    projected_features = project_features_to_srs(ground_truth_data, img_srs)

    # convert projected features to contours
    contours = [polygon_to_contour(f['geometry'], img_transform)
                for f in projected_features]
    return contours

print(len(load_contours(train_ground_truth_filename, train_pl_filename)))

# Utility functions: loading an image

NamedBands = namedtuple('NamedBands', 'b, g, r, nir')

def load_bands(filename):
    """Loads a 4-band BGRNir Planet Image file as a list of masked bands.
    
    The masked bands share the same mask, so editing one band mask will
    edit them all."""
    with rasterio.open(filename) as src:
        b, g, r, nir = src.read()
        mask = src.read_masks(1) == 0 # 0 value means the pixel is masked
    
    bands = NamedBands(b=b, g=g, r=r, nir=nir)
    return NamedBands(*[np.ma.array(b, mask=mask)
                        for b in bands])

def get_rgb(named_bands):
    return [named_bands.r, named_bands.g, named_bands.b]

def check_mask(band):
    return '{}/{} masked'.format(band.mask.sum(), band.mask.size)
    
print(check_mask(load_bands(train_pl_filename).r))

# Utility functions: converting an image to reflectance

NamedCoefficients = namedtuple('NamedCoefficients', 'b, g, r, nir')

def read_refl_coefficients(metadata_filename):
    # https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ndvi/ndvi_planetscope.ipynb
    xmldoc = minidom.parse(metadata_filename)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    return NamedCoefficients(b=coeffs[1],
                             g=coeffs[2],
                             r=coeffs[3],
                             nir=coeffs[4])


def to_reflectance(bands, coeffs):
    return NamedBands(*[b.astype(float)*c for b,c in zip(bands, coeffs)])


def load_refl_bands(filename, metadata_filename):
    bands = load_bands(filename)
    coeffs = read_refl_coefficients(metadata_filename)
    return to_reflectance(bands, coeffs)


print(read_refl_coefficients(train_pl_metadata_filename))

# Utility functions: displaying an image

def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
    """Linear scale from old_min to new_min, old_max to new_max.
    
    Values below min/max are allowed in input and output.
    Min/Max values are two data points that are used in the linear scaling.
    """
    #https://en.wikipedia.org/wiki/Normalization_(image_processing)
    return (ndarray - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
# print(linear_scale(np.array([1,2,10,100,256,2560, 2660]), 2, 2560, 0, 256))

def _mask_to_alpha(bands):
#     band = np.atleast_3d(bands)[...,0]
    band = np.atleast_3d(bands[0])
    alpha = np.zeros_like(band)
    alpha[~band.mask] = 1
    return alpha

def _add_alpha_mask(bands):
    return np.dstack([bands, _mask_to_alpha(bands)])

def bands_to_display(bands, alpha=False):
    """Converts a list of 3 bands to a 3-band rgb, normalized array for display."""  
    assert len(bands) in [1,3]
    all_bands = np.dstack(bands)
    old_min = np.percentile(all_bands, 2)
    old_max = np.percentile(all_bands, 98)

    new_min = 0
    new_max = 1
    scaled = [np.clip(_linear_scale(b.astype(np.double),
                                    old_min, old_max, new_min, new_max),
                      new_min, new_max)
              for b in bands]

    filled = [b.filled(fill_value=new_min) for b in scaled]
    if alpha:
        filled.append(_mask_to_alpha(scaled))

    return np.dstack(filled)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))
for ax in (ax1, ax2):
    ax.set_adjustable('box-forced')

ax1.imshow(bands_to_display(get_rgb(load_bands(train_pl_filename)), alpha=False))
ax1.set_title('Display Bands, No Alpha')

ax2.imshow(bands_to_display(get_rgb(load_bands(train_pl_filename)), alpha=True))
ax2.set_title('Display Bands, Alpha from Mask')
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))
for ax in (ax1, ax2):
    ax.set_adjustable('box-forced')
    
ax1.imshow(bands_to_display(get_rgb(load_refl_bands(train_pl_filename, train_pl_metadata_filename)), alpha=False))
ax1.set_title('Reflectance Bands, No Alpha')

ax2.imshow(bands_to_display(get_rgb(load_refl_bands(train_pl_filename, train_pl_metadata_filename)), alpha=True))
ax2.set_title('Reflectance Bands, Alpha from Mask')
plt.tight_layout()

# Utility functions: displaying contours

def draw_contours(img, contours, color=(0, 1, 0), thickness=2):
    """Draw contours over a copy of the image"""
    n_img = img.copy()
    # http://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html#drawcontours
    cv2.drawContours(n_img,contours,-1,color,thickness=thickness)
    return n_img

plt.figure(1, figsize=(10,10))
plt.imshow(draw_contours(bands_to_display(get_rgb(load_bands(train_pl_filename)), alpha=False),
                         load_contours(train_ground_truth_filename, train_pl_filename)))
_ = plt.title('Contours Drawn over Image')

# Utility functions: masking pixels using contours

def _create_contour_mask(contours, shape):
    """Masks out all pixels that are not within a contour"""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (1), thickness=-1)
    return mask < 1

def combine_masks(masks):
    """Masks any pixel that is masked in any mask.
    
    masks is a list of masks, all the same size"""
    return np.any(np.dstack(masks), 2)

def _add_mask(bands, mask):
    # since band masks are linked, could use any band here
    bands[0].mask = combine_masks([bands[0].mask, mask])

def mask_contours(bands, contours, in_contours=False):
    contour_mask = _create_contour_mask(contours, bands[0].mask.shape)
    if in_contours:
        contour_mask = ~contour_mask
    _add_mask(bands, contour_mask)
    return bands

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))
for ax in (ax1, ax2):
    ax.set_adjustable('box-forced')

ax1.imshow(bands_to_display(get_rgb(mask_contours(load_bands(train_pl_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_pl_filename))),
                            alpha=True))
ax1.set_title('Crop Pixels')

ax2.imshow(bands_to_display(get_rgb(mask_contours(load_bands(train_pl_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_pl_filename),
                                                  in_contours=True)),
                            alpha=True))
ax2.set_title('Non-Crop Pixels')
plt.tight_layout()

def classified_band_from_masks(band_mask, class_masks):
    class_band = np.zeros(band_mask.shape, dtype=np.uint8)

    for i, mask in enumerate(class_masks):
        class_band[~mask] = i

    # turn into masked array, using band_mask as mask
    return np.ma.array(class_band, mask=band_mask)


def create_contour_classified_band(pl_filename, ground_truth_filename):
    band_mask = load_bands(pl_filename)[0].mask
    contour_mask = _create_contour_mask(load_contours(ground_truth_filename, pl_filename),
                                        band_mask.shape)
    return classified_band_from_masks(band_mask, [contour_mask, ~contour_mask])

plt.figure(1, figsize=(10,10))
plt.imshow(create_contour_classified_band(train_pl_filename, train_ground_truth_filename))
_ = plt.title('Ground Truth Classes')

def to_X(bands):
    """Convert to a list of pixel values, maintaining order and filtering out masked pixels."""
    return np.stack([b.compressed() for b in bands], axis=1)

def to_y(classified_band):
    return classified_band.compressed()

def save_cross_val_data(pl_filename, ground_truth_filename, metadata_filename):
    train_class_band = create_contour_classified_band(pl_filename, ground_truth_filename)
    X = to_X(load_refl_bands(pl_filename, metadata_filename))
    y = to_y(train_class_band)

    cross_val_dir = os.path.join('data', 'knn_cross_val')
    if not os.path.exists(cross_val_dir):
        os.makedirs(cross_val_dir)

    xy_file = os.path.join(cross_val_dir, 'xy_file.npz')
    np.savez(xy_file, X=X, y=y)
    return xy_file

print(save_cross_val_data(train_pl_filename,
                          train_ground_truth_filename,
                          train_pl_metadata_filename))

# best n_neighbors found in KNN Parameter Tuning
n_neighbors = 3

def fit_classifier(pl_filename, ground_truth_filename, metadata_filename, n_neighbors):
    n_neighbors = 15
    weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    
    train_class_band = create_contour_classified_band(pl_filename, ground_truth_filename)
    X = to_X(load_refl_bands(pl_filename, metadata_filename))
    y = to_y(train_class_band)
    clf.fit(X, y)
    return clf

clf = fit_classifier(train_pl_filename,
                     train_ground_truth_filename,
                     train_pl_metadata_filename,
                     n_neighbors)

def classified_band_from_y(band_mask, y):
    class_band = np.ma.array(np.zeros(band_mask.shape),
                             mask=band_mask.copy())
    class_band[~class_band.mask] = y
    return class_band


def predict(pl_filename, metadata_filename, clf):
    bands = load_refl_bands(pl_filename, metadata_filename)
    X = to_X(bands)

    y = clf.predict(X)
    
    return classified_band_from_y(bands[0].mask, y)

# it takes a while to run the prediction so cache the results
train_predicted_class_band = predict(train_pl_filename, train_pl_metadata_filename, clf)

def imshow_class_band(ax, class_band):
    """Show classified band with legend. Alters ax in place."""
    im = ax.imshow(class_band)

    # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    colors = [ im.cmap(im.norm(value)) for value in (0,1)]
    labels = ('crop', 'non-crop')

    # https://matplotlib.org/users/legend_guide.html
    # tag: #creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
    patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, labels)]
    
    ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)

def plot_predicted_vs_truth(predicted_class_band, truth_class_band, figsize=(15,15)):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, sharey=True,
                                   figsize=figsize)
    for ax in (ax1, ax2):
        ax.set_adjustable('box-forced')

    imshow_class_band(ax1, predicted_class_band)
    ax1.set_title('Classifier Predicted Classes')

    imshow_class_band(ax2, truth_class_band)
    ax2.set_title('Ground Truth Classes')
    plt.tight_layout()

plot_predicted_vs_truth(train_predicted_class_band,
                        create_contour_classified_band(train_pl_filename,
                                                       train_ground_truth_filename))

# Utility functions: Comparing ground truth and actual in a region

def calculate_ndvi(bands):
    return (bands.nir.astype(np.float) - bands.r) / (bands.nir + bands.r)

def plot_region_compare(region_slice, predicted_class_band, class_band, img_bands):
    # x[1:10:5,::-1] is equivalent to 
    # obj = (slice(1,10,5), slice(None,None,-1)); x[obj]
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                                                 figsize=(10,10))
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_adjustable('box-forced')

    imshow_class_band(ax1, predicted_class_band[region_slice])
    ax1.set_title('Predicted Classes')
    
    imshow_class_band(ax2, class_band[region_slice])
    ax2.set_title('Ground Truth Classes')

    region_bands = NamedBands(*[b[region_slice] for b in img_bands])

    ax3.imshow(bands_to_display(get_rgb(region_bands)))
    ax3.set_title('Planet Scene RGB')

    ax4.imshow(bands_to_display(3*[calculate_ndvi(region_bands)], alpha=False))
    ax4.set_title('Planet Scene NDVI')
    
    plt.tight_layout()

plot_region_compare((slice(1650,2150,None), slice(600,1100,None)),
                    train_predicted_class_band,
                    create_contour_classified_band(train_pl_filename,
                                                   train_ground_truth_filename),
                    load_refl_bands(train_pl_filename, train_pl_metadata_filename))

plt.figure(1, figsize=(8,8))
plt.imshow(bands_to_display(get_rgb(load_refl_bands(test_pl_filename, test_pl_metadata_filename)),
                            alpha=True))
_ = plt.title('Test Image RGB')

test_predicted_class_band = predict(test_pl_filename, test_pl_metadata_filename, clf)

plot_predicted_vs_truth(test_predicted_class_band,
                        create_contour_classified_band(test_pl_filename,
                                                       test_ground_truth_filename))

plot_region_compare((slice(600,1000,None), slice(200,600,None)),
                    test_predicted_class_band,
                    create_contour_classified_band(test_pl_filename,
                                                   test_ground_truth_filename),
                    load_refl_bands(test_pl_filename, test_pl_metadata_filename))

plot_region_compare((slice(200,600,None), slice(400,800,None)),
                    test_predicted_class_band,
                    create_contour_classified_band(test_pl_filename,
                                                   test_ground_truth_filename),
                    load_refl_bands(test_pl_filename, test_pl_metadata_filename))

# train dataset
print(classification_report(to_y(create_contour_classified_band(train_pl_filename,
                                          train_ground_truth_filename)),
                            to_y(train_predicted_class_band),
                            target_names=['crop', 'non-crop']))

# test dataset
print(classification_report(to_y(create_contour_classified_band(test_pl_filename,
                                          test_ground_truth_filename)),
                            to_y(test_predicted_class_band),
                            target_names=['crop', 'non-crop']))

class_band = test_predicted_class_band.astype(np.uint8)

plt.figure(1, figsize=(8,8))
plt.imshow(class_band)
_ = plt.title('Test Predicted Classes')

def denoise(img, plot=False):
    def plot_img(img, title=None):
        if plot:
            figsize = 8; plt.figure(figsize=(figsize,figsize))
            plt.imshow(img)
            if title:
                plt.title(title)
        
    denoised = class_band.copy()
    plot_img(denoised, 'Original')
    
    denoised = cv2.medianBlur(denoised, 5)
    plot_img(denoised, 'Median Blur 5x5')
    
    return denoised

denoised = denoise(class_band, plot=True)

def to_crop_image(class_band):
    denoised = denoise(class_band)
    
    # crop pixels are 1, non-crop pixels are zero
    crop_image = np.zeros(denoised.shape, dtype=np.uint8)
    crop_image[(~class_band.mask) & (denoised == 0)] = 1
    return crop_image

crop_image = to_crop_image(class_band)
plt.figure(figsize=(8,8)); plt.imshow(crop_image)
_ = plt.title('Crop regions set to 1 (yellow)')

def get_crop_contours(img):
    """Identify contours that represent crop segments from a binary image.
    
    The crop pixel values must be 1, non-crop pixels zero.
    """
    # http://docs.opencv.org/trunk/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def is_region(contour):
        num_points = contour.shape[0]
        return num_points >= 3

    return [c for c in contours
            if is_region(c)]

contours = get_crop_contours(crop_image)
print(len(contours))

figsize = 8; plt.figure(figsize=(figsize,figsize))
plt.imshow(draw_contours(bands_to_display(3*[np.ma.array(denoised, mask=class_band.mask)],
                                          alpha=False),
                         contours,
                         thickness=4))
_ = plt.title('Segmented Crop Outlines')

figsize = 8; plt.figure(figsize=(figsize,figsize))
plt.imshow(draw_contours(bands_to_display(get_rgb(load_refl_bands(test_pl_filename,
                                                                  test_pl_metadata_filename)),
                                          alpha=False),
                         contours,
                         thickness=-1))
_ = plt.title('Segmented Crop Regions')

# Utility functions: contours to features

def contour_to_polygon(contour, image_transform):
    """Convert an OpenCV contour to a rasterio Polygon.
    
    image_transform is typically obtained from `img.transform` where 
    img is obtained from `rasterio.open()
    
    :param contour: OpenCV contour
    :param rasterio.Affine image_transform: image transformation"""
    
    # get list of x and y coordinates from contour
    # contour: np.array(<list of points>, dtype=np.int32)
    # contour shape: (<number of points>, 1, 2)
    # point: (col, row)
    points_shape = contour.shape
    
    # get rid of the 1-element 2nd axis then convert 2d shape to a list
    cols_rows = contour.reshape((points_shape[0],points_shape[2])).tolist()
    cols, rows = zip(*cols_rows) # inverse of zip
    
    # convert rows/cols to x/y
    offset = 'ul' # OpenCV: contour point [0,0] is the upper-left corner of pixel 0,0
    points_x, points_y = rasterio.transform.xy(image_transform, rows, cols, offset=offset)
    
    # create geometry from series of points
    points_xy = zip(points_x, points_y)
    return Polygon(points_xy)


def project_features_from_srs(features, img_srs, dst_srs='epsg:4326'):
    """Project features from img_srs.
    
    If dst_srs is not specified, WGS84 (only geojson-supported crs) is assumed.
    
    :param features: list of geojson features to be projected
    :param str img_srs: source spatial reference system
    :param str dst_srs: destination spatial reference system
    """
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init=img_srs),
        pyproj.Proj(init=dst_srs))
    
    return [project_feature(f, proj_fcn) for f in features]


def polygon_to_feature(polygon):
    return {'type': 'Feature',
            'properties': {},
            'geometry': mapping(polygon)}    


def contours_to_features(contours, pl_filename):
    """Convert contours to geojson features.
    
    Features are simplified with a tolerance of 3 pixels then
    filtered to those with 100x100m2 area.
    """
    with rasterio.open(test_pl_filename) as img:
        img_srs = img.crs['init']
        img_transform = img.transform
        
    polygons = [contour_to_polygon(c, img_transform)
                for c in contours]

    tolerance = 3
    simplified_polygons = [p.simplify(tolerance, preserve_topology=True)
                           for p in polygons]

    def passes_filters(p):
        min_area = 50 * 50
        return p.area > min_area

    features = [polygon_to_feature(p)
                for p in simplified_polygons
                if passes_filters(p)]

    return project_features_from_srs(features, img_srs)

features = contours_to_features(contours, test_pl_filename)
print(len(features))

# Create crop feature layer
feature_collection = {
    "type": "FeatureCollection",
    "features": features,
    "properties": {"style":{
        'weight': 0,
        'fillColor': "blue",
        'fillOpacity': 1}}
}

feature_layer = ipyl.GeoJSON(data=feature_collection)

# Initialize map using parameters from above map
# and deleting map instance if it exists
try:
    del crop_map
except NameError:
    pass


zoom = 13
center = [38.30839, -121.55187] # lat/lon

# Create map, adding box drawing controls
# Reuse parameters if map already exists
try:
    center = crop_map.center
    zoom = crop_map.zoom
except NameError:
    pass

# Change tile layer to one that makes it easier to see crop features
# Layer selected using https://leaflet-extras.github.io/leaflet-providers/preview/
map_tiles = ipyl.TileLayer(url='http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png')
crop_map = ipyl.Map(
        center=center, 
        zoom=zoom,
        default_tiles = map_tiles
    )

crop_map.add_layer(feature_layer)  


# Display map
crop_map



