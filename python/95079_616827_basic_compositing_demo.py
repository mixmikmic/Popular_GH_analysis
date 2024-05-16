get_ipython().system('gdalinfo data/175322.tif ')
get_ipython().system('gdalinfo data/175323.tif')
get_ipython().system('gdalinfo data/175325.tif')

get_ipython().system('gdal_merge.py -v data/175322.tif data/175323.tif data/175325.tif -o output/mtdana-merged.tif')

get_ipython().system('gdalinfo output/mtdana-merged.tif')

get_ipython().system('gdal_translate -of "PNG" -outsize 10% 0% output/mtdana-merged.tif output/mtdana-merged.png')

from IPython.display import Image
Image(filename="output/mtdana-merged.png")

get_ipython().system('gdalwarp -of GTiff -cutline data/mt-dana-small.geojson -crop_to_cutline output/mtdana-merged.tif output/mtdana-cropped.tif')

get_ipython().system('gdal_translate -of "PNG" -outsize 10% 0% output/mtdana-cropped.tif output/mtdana-cropped.png')

from IPython.display import Image
Image(filename="output/mtdana-cropped.png")

