import urllib.request, urllib.parse, urllib.error
import json

with open('APIkeys.json') as f:
    keys = json.load(f)
    weatherapi = keys['weatherapi']

serviceurl = 'http://api.openweathermap.org/data/2.5/weather?'
apikey = 'APPID='+weatherapi

while True:
    address = input('Enter the name of a town (enter \'quit\' or hit ENTER to quit): ')
    if len(address) < 1 or address=='quit': break

    url = serviceurl + urllib.parse.urlencode({'q': address})+'&'+apikey
    print(f'Retrieving the weather data of {address} now... ')
    uh = urllib.request.urlopen(url)
    
    data = uh.read()
    json_data=json.loads(data)
    
    main=json_data['main']
    description = json_data['weather'][-1]['description']
    
    pressure_mbar = main['pressure']
    pressure_inch_Hg = pressure_mbar*0.02953
    humidity = main['humidity']
    temp_min = main['temp_min']-273
    temp_max = main['temp_max']-273
    temp = main['temp']-273
    
    print(f"\nRight now {address} has {description}. Key weather parameters are as follows\n"+"-"*70)
    print(f"Pressure: {pressure} mbar/{pressure_inch_Hg} inch Hg")
    print(f"Humidity: {humidity}%")
    print(f"Temperature: {round(temp,2)} degree C")



