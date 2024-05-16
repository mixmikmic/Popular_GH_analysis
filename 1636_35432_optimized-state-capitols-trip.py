# https://en.wikipedia.org/wiki/List_of_state_capitols_in_the_United_States

all_waypoints = ['Alabama State Capitol, 600 Dexter Avenue, Montgomery, AL 36130',
                 #'Alaska State Capitol, Juneau, AK',
                 'Arizona State Capitol, 1700 W Washington St, Phoenix, AZ 85007',
                 'Arkansas State Capitol, 500 Woodlane Street, Little Rock, AR 72201',
                 'L St & 10th St, Sacramento, CA 95814',
                 '200 E Colfax Ave, Denver, CO 80203',
                 'Connecticut State Capitol, 210 Capitol Ave, Hartford, CT 06106',
                 'Legislative Hall: The State Capitol, Legislative Avenue, Dover, DE 19901',
                 '402 S Monroe St, Tallahassee, FL 32301',
                 'Georgia State Capitol, Atlanta, GA 30334',
                 #'Hawaii State Capitol, 415 S Beretania St, Honolulu, HI 96813'
                 '700 W Jefferson St, Boise, ID 83720',
                 'Illinois State Capitol, Springfield, IL 62756',
                 'Indiana State Capitol, Indianapolis, IN 46204',
                 'Iowa State Capitol, 1007 E Grand Ave, Des Moines, IA 50319',
                 '300 SW 10th Ave, Topeka, KS 66612',
                 'Kentucky State Capitol Building, 700 Capitol Avenue, Frankfort, KY 40601',
                 'Louisiana State Capitol, Baton Rouge, LA 70802',
                 'Maine State House, Augusta, ME 04330',
                 'Maryland State House, 100 State Cir, Annapolis, MD 21401',
                 'Massachusetts State House, Boston, MA 02108',
                 'Michigan State Capitol, Lansing, MI 48933',
                 'Minnesota State Capitol, St Paul, MN 55155',
                 '400-498 N West St, Jackson, MS 39201',
                 'Missouri State Capitol, Jefferson City, MO 65101',
                 'Montana State Capitol, 1301 E 6th Ave, Helena, MT 59601',
                 'Nebraska State Capitol, 1445 K Street, Lincoln, NE 68509',
                 'Nevada State Capitol, Carson City, NV 89701',
                 'State House, 107 North Main Street, Concord, NH 03303',
                 'New Jersey State House, Trenton, NJ 08608',
                 'New Mexico State Capitol, Santa Fe, NM 87501',
                 'New York State Capitol, State St. and Washington Ave, Albany, NY 12224',
                 'North Carolina State Capitol, Raleigh, NC 27601',
                 'North Dakota State Capitol, Bismarck, ND 58501',
                 'Ohio State Capitol, 1 Capitol Square, Columbus, OH 43215',
                 'Oklahoma State Capitol, Oklahoma City, OK 73105',
                 'Oregon State Capitol, 900 Court St NE, Salem, OR 97301',
                 'Pennsylvania State Capitol Building, North 3rd Street, Harrisburg, PA 17120',
                 'Rhode Island State House, 82 Smith Street, Providence, RI 02903',
                 'South Carolina State House, 1100 Gervais Street, Columbia, SC 29201',
                 '500 E Capitol Ave, Pierre, SD 57501',
                 'Tennessee State Capitol, 600 Charlotte Avenue, Nashville, TN 37243',
                 'Texas Capitol, 1100 Congress Avenue, Austin, TX 78701',
                 'Utah State Capitol, Salt Lake City, UT 84103',
                 'Vermont State House, 115 State Street, Montpelier, VT 05633',
                 'Virginia State Capitol, Richmond, VA 23219',
                 'Washington State Capitol Bldg, 416 Sid Snyder Ave SW, Olympia, WA 98504',
                 'West Virginia State Capitol, Charleston, WV 25317',
                 '2 E Main St, Madison, WI 53703',
                 'Wyoming State Capitol, Cheyenne, WY 82001']

len(all_waypoints)

import googlemaps

gmaps = googlemaps.Client(key='ENTER YOUR GOOGLE MAPS KEY HERE')

from itertools import combinations

waypoint_distances = {}
waypoint_durations = {}

for (waypoint1, waypoint2) in combinations(all_waypoints, 2):
    try:
        route = gmaps.distance_matrix(origins=[waypoint1],
                                      destinations=[waypoint2],
                                      mode='driving', # Change this to 'walking' for walking directions,
                                                      # 'bicycling' for biking directions, etc.
                                      language='English',
                                      units='metric')

        # 'distance' is in meters
        distance = route['rows'][0]['elements'][0]['distance']['value']

        # 'duration' is in seconds
        duration = route['rows'][0]['elements'][0]['duration']['value']

        waypoint_distances[frozenset([waypoint1, waypoint2])] = distance
        waypoint_durations[frozenset([waypoint1, waypoint2])] = duration
    
    except Exception as e:
        print('Error with finding the route between {} and {}.'.format(waypoint1, waypoint2))

with open('my-waypoints-dist-dur.tsv', 'w') as out_file:
    out_file.write('\t'.join(['waypoint1',
                              'waypoint2',
                              'distance_m',
                              'duration_s']))
    
    for (waypoint1, waypoint2) in waypoint_distances.keys():
        out_file.write('\n' +
                       '\t'.join([waypoint1,
                                  waypoint2,
                                  str(waypoint_distances[frozenset([waypoint1, waypoint2])]),
                                  str(waypoint_durations[frozenset([waypoint1, waypoint2])])]))

import pandas as pd
import numpy as np

waypoint_distances = {}
waypoint_durations = {}
all_waypoints = set()

waypoint_data = pd.read_csv('my-waypoints-dist-dur.tsv', sep='\t')

for i, row in waypoint_data.iterrows():
    # Distance = meters
    waypoint_distances[frozenset([row.waypoint1, row.waypoint2])] = row.distance_m
    
    # Duration = hours
    waypoint_durations[frozenset([row.waypoint1, row.waypoint2])] = row.duration_s / (60. * 60.)
    all_waypoints.update([row.waypoint1, row.waypoint2])

import random
import numpy as np
import copy
from tqdm import tqdm

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
creator.create('Individual', list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register('waypoints', random.sample, all_waypoints, random.randint(2, 20))
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.waypoints)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def eval_capitol_trip(individual):
    """
        This function returns the total distance traveled on the current road trip
        as well as the number of waypoints visited in the trip.
        
        The genetic algorithm will favor road trips that have shorter
        total distances traveled and more waypoints visited.
    """
    trip_length = 0.
    individual = list(individual)
    
    # Adding the starting point to the end of the trip forces it to be a round-trip
    individual += [individual[0]]
    
    for index in range(1, len(individual)):
        waypoint1 = individual[index - 1]
        waypoint2 = individual[index]
        trip_length += waypoint_distances[frozenset([waypoint1, waypoint2])]
        
    return len(set(individual)), trip_length

def pareto_selection_operator(individuals, k):
    """
        This function chooses what road trips get copied into the next generation.
        
        The genetic algorithm will favor road trips that have shorter
        total distances traveled and more waypoints visited.
    """
    return tools.selNSGA2(individuals, int(k / 5.)) * 5

def mutation_operator(individual):
    """
        This function applies a random change to one road trip:
        
            - Insert: Adds one new waypoint to the road trip
            - Delete: Removes one waypoint from the road trip
            - Point: Replaces one waypoint with another different one
            - Swap: Swaps the places of two waypoints in the road trip
    """
    possible_mutations = ['swap']
    
    if len(individual) < len(all_waypoints):
        possible_mutations.append('insert')
        possible_mutations.append('point')
    if len(individual) > 2:
        possible_mutations.append('delete')
    
    mutation_type = random.sample(possible_mutations, 1)[0]
    
    # Insert mutation
    if mutation_type == 'insert':
        waypoint_to_add = individual[0]
        while waypoint_to_add in individual:
            waypoint_to_add = random.sample(all_waypoints, 1)[0]
            
        index_to_insert = random.randint(0, len(individual) - 1)
        individual.insert(index_to_insert, waypoint_to_add)
    
    # Delete mutation
    elif mutation_type == 'delete':
        index_to_delete = random.randint(0, len(individual) - 1)
        del individual[index_to_delete]
    
    # Point mutation
    elif mutation_type == 'point':
        waypoint_to_add = individual[0]
        while waypoint_to_add in individual:
            waypoint_to_add = random.sample(all_waypoints, 1)[0]
        
        index_to_replace = random.randint(0, len(individual) - 1)
        individual[index_to_replace] = waypoint_to_add
        
    # Swap mutation
    elif mutation_type == 'swap':
        index1 = random.randint(0, len(individual) - 1)
        index2 = index1
        while index2 == index1:
            index2 = random.randint(0, len(individual) - 1)
            
        individual[index1], individual[index2] = individual[index2], individual[index1]
    
    return individual,


toolbox.register('evaluate', eval_capitol_trip)
toolbox.register('mutate', mutation_operator)
toolbox.register('select', pareto_selection_operator)

def pareto_eq(ind1, ind2):
    return np.all(ind1.fitness.values == ind2.fitness.values)

pop = toolbox.population(n=1000)
hof = tools.ParetoFront(similar=pareto_eq)
stats = tools.Statistics(lambda ind: (int(ind.fitness.values[0]), round(ind.fitness.values[1], 2)))
stats.register('Minimum', np.min, axis=0)
stats.register('Maximum', np.max, axis=0)
# This stores a copy of the Pareto front for every generation of the genetic algorithm
stats.register('ParetoFront', lambda x: copy.deepcopy(hof))
# This is a hack to make the tqdm progress bar work
stats.register('Progress', lambda x: pbar.update())

# How many iterations of the genetic algorithm to run
# The more iterations you allow it to run, the better the solutions it will find
total_gens = 5000

pbar = tqdm(total=total_gens)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0., mutpb=1.0, ngen=total_gens, 
                               stats=stats, halloffame=hof, verbose=False)
pbar.close()

def create_animated_road_trip_map(optimized_routes):
    """
        This function takes a list of optimized road trips and generates
        an animated map of them using the Google Maps API.
    """
    
    # This line makes the road trips round trips
    optimized_routes = [list(route) + [route[0]] for route in optimized_routes]

    Page_1 = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
        <meta name="description" content="Randy Olson uses machine learning to find the optimal road trip across the U.S.">
        <meta name="author" content="Randal S. Olson">
        
        <title>An optimized road trip across the U.S. according to machine learning</title>
        <style>
          html, body, #map-canvas {
              height: 100%;
              margin: 0px;
              padding: 0px
          }
          #panel {
              position: absolute;
              top: 5px;
              left: 50%;
              margin-left: -180px;
              z-index: 5;
              background-color: #fff;
              padding: 10px;
              border: 1px solid #999;
          }
        </style>
        <script src="https://maps.googleapis.com/maps/api/js?v=3"></script>
        <script>
            var routesList = [];
            var markerOptions = {icon: "http://maps.gstatic.com/mapfiles/markers2/marker.png"};
            var directionsDisplayOptions = {preserveViewport: true,
                                            markerOptions: markerOptions};
            var directionsService = new google.maps.DirectionsService();
            var map;
            var mapNum = 0;
            var numRoutesRendered = 0;
            var numRoutes = 0;
            
            function initialize() {
                var center = new google.maps.LatLng(39, -96);
                var mapOptions = {
                    zoom: 5,
                    center: center
                };
                map = new google.maps.Map(document.getElementById("map-canvas"), mapOptions);
                for (var i = 0; i < routesList.length; i++) {
                    routesList[i].setMap(map); 
                }
            }
            function calcRoute(start, end, routes) {
                var directionsDisplay = new google.maps.DirectionsRenderer(directionsDisplayOptions);
                var waypts = [];
                for (var i = 0; i < routes.length; i++) {
                    waypts.push({
                        location:routes[i],
                        stopover:true});
                    }

                var request = {
                    origin: start,
                    destination: end,
                    waypoints: waypts,
                    optimizeWaypoints: false,
                    travelMode: google.maps.TravelMode.DRIVING
                };
                directionsService.route(request, function(response, status) {
                    if (status == google.maps.DirectionsStatus.OK) {
                        directionsDisplay.setDirections(response);
                        directionsDisplay.setMap(map);
                        numRoutesRendered += 1;
                        
                        if (numRoutesRendered == numRoutes) {
                            mapNum += 1;
                            if (mapNum < 47) {
                                setTimeout(function() {
                                    return createRoutes(allRoutes[mapNum]);
                                }, 5000);
                            }
                        }
                    }
                });
                
                routesList.push(directionsDisplay);
            }
            function createRoutes(route) {
                // Clear the existing routes (if any)
                for (var i = 0; i < routesList.length; i++) {
                    routesList[i].setMap(null);
                }
                routesList = [];
                numRoutes = Math.floor((route.length - 1) / 9 + 1);
                numRoutesRendered = 0;
            
                // Google's free map API is limited to 10 waypoints so need to break into batches
                var subset = 0;
                while (subset < route.length) {
                    var waypointSubset = route.slice(subset, subset + 10);
                    var startPoint = waypointSubset[0];
                    var midPoints = waypointSubset.slice(1, waypointSubset.length - 1);
                    var endPoint = waypointSubset[waypointSubset.length - 1];
                    calcRoute(startPoint, endPoint, midPoints);
                    subset += 9;
                }
            }
            
            allRoutes = [];
            """
    Page_2 = """
            createRoutes(allRoutes[mapNum]);
            google.maps.event.addDomListener(window, "load", initialize);
        </script>
      </head>
      <body>
        <div id="map-canvas"></div>
      </body>
    </html>
    """

    with open('us-state-capitols-animated-map.html', 'w') as output_file:
        output_file.write(Page_1)
        for route in optimized_routes:
            output_file.write('allRoutes.push({});'.format(str(route)))
        output_file.write(Page_2)

create_animated_road_trip_map(reversed(hof))

get_ipython().system('open us-state-capitols-animated-map.html')

def create_individual_road_trip_maps(optimized_routes):
    """
        This function takes a list of optimized road trips and generates
        individual maps of them using the Google Maps API.
    """
    
    # This line makes the road trips round trips
    optimized_routes = [list(route) + [route[0]] for route in optimized_routes]

    for route_num, route in enumerate(optimized_routes):
        Page_1 = """
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
            <meta name="description" content="Randy Olson uses machine learning to find the optimal road trip across the U.S.">
            <meta name="author" content="Randal S. Olson">

            <title>An optimized road trip across the U.S. according to machine learning</title>
            <style>
              html, body, #map-canvas {
                  height: 100%;
                  margin: 0px;
                  padding: 0px
              }
              #panel {
                  position: absolute;
                  top: 5px;
                  left: 50%;
                  margin-left: -180px;
                  z-index: 5;
                  background-color: #fff;
                  padding: 10px;
                  border: 1px solid #999;
              }
            </style>
            <script src="https://maps.googleapis.com/maps/api/js?v=3"></script>
            <script>
                var routesList = [];
                var markerOptions = {icon: "http://maps.gstatic.com/mapfiles/markers2/marker.png"};
                var directionsDisplayOptions = {preserveViewport: true,
                                                markerOptions: markerOptions};
                var directionsService = new google.maps.DirectionsService();
                var map;

                function initialize() {
                    var center = new google.maps.LatLng(39, -96);
                    var mapOptions = {
                        zoom: 5,
                        center: center
                    };
                    map = new google.maps.Map(document.getElementById("map-canvas"), mapOptions);
                    for (var i = 0; i < routesList.length; i++) {
                        routesList[i].setMap(map); 
                    }
                }
                function calcRoute(start, end, routes) {
                    var directionsDisplay = new google.maps.DirectionsRenderer(directionsDisplayOptions);
                    var waypts = [];
                    for (var i = 0; i < routes.length; i++) {
                        waypts.push({
                            location:routes[i],
                            stopover:true});
                        }

                    var request = {
                        origin: start,
                        destination: end,
                        waypoints: waypts,
                        optimizeWaypoints: false,
                        travelMode: google.maps.TravelMode.DRIVING
                    };
                    directionsService.route(request, function(response, status) {
                        if (status == google.maps.DirectionsStatus.OK) {
                            directionsDisplay.setDirections(response);
                            directionsDisplay.setMap(map);
                        }
                    });

                    routesList.push(directionsDisplay);
                }
                function createRoutes(route) {
                    // Google's free map API is limited to 10 waypoints so need to break into batches
                    var subset = 0;
                    while (subset < route.length) {
                        var waypointSubset = route.slice(subset, subset + 10);
                        var startPoint = waypointSubset[0];
                        var midPoints = waypointSubset.slice(1, waypointSubset.length - 1);
                        var endPoint = waypointSubset[waypointSubset.length - 1];
                        calcRoute(startPoint, endPoint, midPoints);
                        subset += 9;
                    }
                }

                """
        Page_2 = """
                createRoutes(optimized_route);
                google.maps.event.addDomListener(window, "load", initialize);
            </script>
          </head>
          <body>
            <div id="map-canvas"></div>
          </body>
        </html>
        """

        with open('optimized-us-capitol-trip-{}-states.html'.format(route_num + 2), 'w') as output_file:
            output_file.write(Page_1)
            output_file.write('optimized_route = {};'.format(str(route)))
            output_file.write(Page_2)

create_individual_road_trip_maps(reversed(hof))

get_ipython().system('open optimized-us-capitol-trip-48-states.html')

