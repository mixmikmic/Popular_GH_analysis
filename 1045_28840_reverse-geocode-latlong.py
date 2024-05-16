# import necessary modules
import pandas as pd, requests, logging, time

# magic command to display matplotlib plots inline within the ipython notebook
get_ipython().magic('matplotlib inline')

# configure logging for our tool
lfh = logging.FileHandler('logs/reverse_geocoder.log', mode='w', encoding='utf-8')
lfh.setFormatter(logging.Formatter('%(levelname)s %(asctime)s %(message)s'))
log = logging.getLogger('reverse_geocoder')
log.setLevel(logging.INFO)
log.addHandler(lfh)
log.info('process started')

# load the gps coordinate data
df = pd.read_csv('data/summer-travel-gps-no-city-country.csv', encoding='utf-8')

# create new columns
df['geocode_data'] = ''
df['city'] = ''
df['country'] = ''

df.head()

# function that handles the geocoding requests
def reverse_geocode(latlng):
    time.sleep(0.1)
    url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng={0}'    
    request = url.format(latlng)
    log.info(request)
    response = requests.get(request)
    data = response.json()
    if 'results' in data and len(data['results']) > 0:
        return data['results'][0]

# create concatenated lat+lng column then reverse geocode each value
df['latlng'] = df.apply(lambda row: '{},{}'.format(row['lat'], row['lon']), axis=1)
df['geocode_data'] = df['latlng'].map(reverse_geocode)
df.head()

# identify municipality and country data in the json that google sent back
def parse_city(geocode_data):
    if (not geocode_data is None) and ('address_components' in geocode_data):
        for component in geocode_data['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
            elif 'postal_town' in component['types']:
                return component['long_name']
            elif 'administrative_area_level_2' in component['types']:
                return component['long_name']
            elif 'administrative_area_level_1' in component['types']:
                return component['long_name']
    return None

def parse_country(geocode_data):
    if (not geocode_data is None) and ('address_components' in geocode_data):
        for component in geocode_data['address_components']:
            if 'country' in component['types']:
                return component['long_name']
    return None

df['city'] = df['geocode_data'].map(parse_city)
df['country'] = df['geocode_data'].map(parse_country)
print(len(df))
df.head()

# google's geocoder fails on anything in kosovo, so do those manually now
df.loc[df['country']=='', 'country'] = 'Kosovo'
df.loc[df['city']=='', 'city'] = 'Prizren'

# save our reverse-geocoded data set
df.to_csv('data/summer-travel-gps-full.csv', encoding='utf-8', index=False)



