# This example inspired by a post by Nathan Yau on FlowingData
# http://flowingdata.com/2009/11/12/how-to-make-a-us-county-thematic-map-using-free-tools/

import requests

resp = requests.get("http://upload.wikimedia.org/wikipedia/commons/5/5f/USA_Counties_with_FIPS_and_names.svg")
usa = resp.content

from IPython.display import display_svg

with open('usa.svg', 'wb') as fout:
    fout.write(usa)

# Display the original image. This is an SVG so it can scale seamlessly

# The data file on the website http://www.bls.gov/lau/#tables
# spans the years 1990 to 2013, so we can simply change the 13 
# in the URL to a different value to get data/plot for a different year.

display_svg(usa, raw=True)

# The data file on the website http://www.bls.gov/lau/#tables
# spans the years 1990 to 2013, so we can simply change the 13 
# in the URL to a different value to get data/plot for a different year.

# Grab and parse the Labor data by using Pandas

import pandas as pd 

# We specify our Column Names

headers = ['LAUS Code', 'StateFIPS', 'CountyFIPS', 'County Name', 'Year',            'Labor Force', 'Employed', 'Unemployed', 'Rate']

# The column widths for fixed width formatting
cs =[(0, 16), (17, 21), (22, 28), (29, 79), (80, 85), (86, 99), (100, 112), (113, 123), (124, 132)]

# The converter functions. We simply use string, we can't use dtypes with a Python engine
cvf = {0 : str, 1 : str, 2: str, 3 : str, 4 : str, 5 : str, 6 : str, 7 : str, 8 : str}

# Read in the data. We skip first five rows that are header info
ud = pd.read_fwf('http://www.bls.gov/lau/laucnty13.txt', converters = cvf, colspecs = cs,                  skiprows=5, header=0, names=headers)

# We drop last three rows that contain footer info.

ud = ud.dropna()

# Now we build a second DataFrame that has the Rate indexed by the FIPS code.
unemployment = pd.DataFrame(ud.Rate.astype(float))
unemployment['FIPS'] = ud.StateFIPS + ud.CountyFIPS
unemployment.set_index('FIPS', inplace = True)

# Test the result

print(unemployment.head(3))
print(unemployment.tail(3))

# Now we turn to parsing the SVG file to modify the styles.

from bs4 import BeautifulSoup

# The SVG file is an XML file so parse appropriately.

soup = BeautifulSoup(usa)

# Find distinct FIPS zones, which are each in a different path.
paths = soup.findAll('path')

# For Color selection, this is a fgreat site:
# http://colorbrewer2.org

# Default FlowingData Map colors
#colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
 
# I Prefer Blues, and more variation.
colors = ['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    
# FlowingData County style (I changed opacity to 0.75 from 1)
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:0.75;' +     'stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;' +     'marker-start:none;stroke-linejoin:bevel;fill:'

# Now iterate through the path elements to modify the path style based on the unemployment rate
for p in paths:
    
    # We need to not try to modify two special paths for State Lines or Separators
    
    if p['id'] not in ['State_lines', 'separator ']:
        try:
            # We simply access our Panda DataFrame
            rate = unemployment.Rate[p['id']]
        except:
            continue
            
        # Now we simply cascade through the unemployemtn rates. Ideally we chance this to a function

        if rate > 11.9:
            color_class = 6
        elif rate > 9.9:
            color_class = 5
        elif rate > 9.9:
            color_class = 4
        elif rate > 7.9:
            color_class = 3
        elif rate > 5.9:
            color_class = 2
        elif rate > 3.9:
            color_class = 1
        else:
            color_class = 0

        # Modify color by our scheme
        color = colors[color_class]
        
        # Now set the nw path style. Ideally again this is built by using string formatting.
        p['style'] = path_style + color
 
# Now our parse tree is done so output it appropriately.
cusa  = soup.prettify()

# And show the result.
display_svg(soup, raw=True)

with open('cusa.svg', 'w') as fout:
    fout.write(cusa)

# Alternative display method. Can use this to compare to different year's values.

from IPython.display import HTML
HTML(cusa)

