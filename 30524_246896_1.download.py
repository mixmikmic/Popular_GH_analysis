import os
from urllib.request import urlretrieve

import requests

# Specify the figshare article ID
figshare_id = 3487685

# Use the figshare API to retrieve article metadata
url = "https://api.figshare.com/v2/articles/{}".format(figshare_id)
response = requests.get(url)
response = response.json()

# Show the version specific DOI
response['doi']

# Make the download directory if it does not exist
if not os.path.exists('download'):
    os.mkdir('download')

# Download the files specified by the metadata
for file_info in response['files']:
    url = file_info['download_url']
    name = file_info['name']
    print('Downloading {} to `{}`'.format(url, name))
    path = os.path.join('download', name)
    urlretrieve(url, path)

