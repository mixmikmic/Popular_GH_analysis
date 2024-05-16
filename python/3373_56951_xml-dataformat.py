import csv

airports = []

with open('/home/data_scientist/rppdm/data/airports.csv', 'r') as csvfile:
    
    for row in csv.reader(csvfile, delimiter=','):
        airports.append(row)

print(airports[0:3])

import html 
import xml.etree.ElementTree as ET

data = '<?xml version="1.0"?>\n' + '<airports>\n'

for airport in airports[1:]:
    data += '    <airport name="{0}">\n'.format(html.escape(airport[1]))
    data += '        <iata>' + str(airport[0]) + '</iata>\n'
    data += '        <city>' + str(airport[2]) + '</city>\n'
    data += '        <state>' + str(airport[3]) + '</state>\n'
    data += '        <country>' + str(airport[4]) + '</country>\n'
    data += '        <latitude>' + str(airport[5]) + '</latitude>\n'
    data += '        <longitude>' + str(airport[6]) + '</longitude>\n'

    data += '    </airport>\n'

data += '</airports>\n'

tree = ET.ElementTree(ET.fromstring(data))


with open('data.xml', 'w') as fout:
    tree.write(fout, encoding='unicode')

get_ipython().system('head -9 data.xml')

data = [["iata", "airport", "city", "state", "country", "lat", "long"]]

tree = ET.parse('data.xml')
root = tree.getroot()

for airport in root.findall('airport'):
    row = []
    row.append(airport[0].text)
    row.append(airport.attrib['name'])
    row.append(airport[1].text)
    row.append(airport[2].text)
    row.append(airport[3].text)
    row.append(airport[4].text)
    row.append(airport[5].text)

    data.append(row)
    
print(data[:5])

