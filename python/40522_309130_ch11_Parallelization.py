from IPython import parallel
engines = parallel.Client()
engines.block = True
print engines.ids

engines[0].execute('a = 2') 
engines[0].execute('b = 10') 
engines[0].execute('c = a + b')  
engines[0].pull('c')

engines[0].execute('a = 2') 
engines[0].execute('b = 10') 
engines[1].execute('a = 9') 
engines[1].execute('b = 7') 
engines[0:2].execute('c = a + b')   
engines[0:2].pull('c')

dview2 = engines[0:2]    

import time
import numpy as np

# Create four 1000x1000 matrix
A0 = np.random.rand(1000,1000)
B0 = np.random.rand(1000,1000)
A1 = np.random.rand(1000,1000)
B1 = np.random.rand(1000,1000)

t0 = time.time() 

C0 = np.dot(A0, B0)
C1 = np.dot(A1, B1)
    
print "Time in seconds (Computations): ", time.time() - t0 

dview2.execute('import numpy as np')       # We import numpy on both engines!

t0 = time.time()
engines[0].push(dict(A=A0, B=B0))    # We send A0 and B0 to engine 0 
engines[1].push(dict(A=A1, B=B1))    # We send A1 and B1 to engine 1 

t0_computations = time.time()

dview2.execute('C = np.dot(A,B)')
    
print "Computations: ", time.time() - t0_computations

[C0, C1] = dview2.pull('C')
print "Time in seconds: ", time.time() - t0

def mul(A, B):
    import numpy as np
    C = np.dot(A, B)
    return C

[C0, C1] = dview2.map(mul,[A0, A1],[B0, B1])

engines[0].execute('my_id = "engineA"') 
engines[1].execute('my_id = "engineB"')

def sleep_and_return_id(sec):     
    import time     
    time.sleep(sec)                      
    return my_id,sec

dview2.map(sleep_and_return_id, [3,3,3,1,1,1])

engines.block = True
lview2 = engines.load_balanced_view(targets=[0,1])

lview2.map(sleep_and_return_id, [3,3,3,1,1,1])

get_ipython().magic('reset -f')

from IPython import parallel
from itertools import islice
from itertools import cycle
from collections import Counter
import sys
import time

#Connect to the Ipython cluster    
engines = parallel.Client()

#Create a DirectView to all engines
dview = engines.direct_view()

print "The number of engines in the cluster is: " + str(len(engines.ids))

get_ipython().run_cell_magic('px', '', '\n# The %%px magic executes the code of this cell on each engine.\n\nfrom datetime import datetime\nfrom collections import Counter\n\nimport pandas as pd\nimport numpy as np\n\n# A Counter object to store engine\'s local result\nlocal_total = Counter();\n\ndef dist(p0, p1):\n    "Returns the distance**2 between two points"\n    # We compute the squared distance. Since we only want to compare\n    # distances there is no need to compute the square root (sqrt) \n    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2\n\n# Coordinates (latitude, longitude) of diferent points of the island\ndistrict_dict = { \n    \'Financial\': [40.724863, -73.994718], \n    \'Midtown\': [40.755905, -73.984997],\n    \'Chinatown\': [40.716224, -73.995925],\n    \'WTC\': [40.711724, -74.012888],\n    \'Harlem\': [40.810469, -73.943318],\n    \'Uppertown\': [40.826381, -73.943964],\n    \'Soho\': [40.723783, -74.001237],\n    \'UpperEastSide\': [40.773861, -73.956329],\n    \'UpperWestSide\': [40.787347, -73.975267]\n    }\n\n# Computes the distance to each district center and obtains the one that\n# gives minimum distance\ndef get_district(coors):\n    "Given a coordinate inn latitude and longitude, returns the district in Manhatan"   \n    #If dist^2 is bigger than 0.0005, the district is \'None\'.\n    dist_min = 0.0005\n    district = None\n    for key in district_dict.iterkeys():\n        d = dist(coors, district_dict[key])\n        if dist_min > d:\n            dist_min = d\n            district = key\n    return district\n\ndef is_morning(d):\n    "Given a datetime, returns if it was on morning or not"\n    h = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").hour\n    return 0 <= h and h < 12\n\ndef is_weekend(d):\n    "Given a datetime, returns if it was on weekend or not"\n    wday = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").weekday() #strptime transforms str to date\n    return 4 < wday <= 6\n\n#Function that classifies a single data\ndef classify(x):\n    "Given a tuple with a datetime, latitude and longitude, returns the group where it fits"\n    date, lat, lon = x\n    latitude = float(lat)\n    longitude = float(lon)\n    return is_weekend(date), is_morning(date), get_district([latitude, longitude])\n\n# Function that given a dictionary (data), applies classify function on each element\n# and returns an histogram in a Counter object\ndef process(b):\n    #Recives a block (list of strings) and updates result in global var local_total()\n    global local_total\n    \n    #Create an empty df. Preallocate the space we need by providing the index (number of rows)\n    df = pd.DataFrame(index=np.arange(0,len(b)), columns=(\'datetime\',\'latitude\',\'longitude\'))\n    \n    # Data is a list of lines, containing datetime at col 5 and latitude at row 11.\n    # Allocate in the dataFrame the datetime and latitude and longitude dor each line in data\n    count = 0\n    for line in b:\n        elements = line.split(",")\n        df.loc[count] = elements[5], elements[11], elements[10]\n        count += 1\n        \n    #Delete NaN values from de DF\n    df.dropna(thresh=(len(df.columns) - 1), axis=0)\n    \n    #Apply classify function to the dataFrame\n    cdf = df.apply(classify, axis=1)\n    \n    #Increment the global variable local_total\n    local_total += Counter(cdf.value_counts().to_dict())\n\n# Initialization function\ndef init():\n    #Reset total var\n    global local_total\n    local_total = Counter()')

# This is the main code executed on the client
t0 = time.time() 

#File to be processed
filename = 'trip_data.csv'

def get_chunk(f,N):
    """ Returns blocks of nl lines from the file descriptor fd"""
    #Deletes first line on first chunk (header line)
    first = 1
    while True:
        new_chunk = list(islice(f, first, N))
        if not new_chunk:
            break
        first = 0
        yield new_chunk

# A simple counter to verify execution
chunk_n = 0

# Number of lines to be sent to each engine at a time. Use carefully!
lines_per_block = 20

# Create an emty list of async tasks. One element for each engine
async_tasks = [None] * len(engines.ids)

# Cycle Object to get an infinite iterator over the list of engines
c_engines = cycle(engines.ids)

# Initialize each engine. Observe that the execute is performed
# in a non-blocking fashion.
for i in engines.ids:
    async_tasks[i] = engines[i].execute('init()', block=False)

# The variable to store results
global_result = Counter()

# Open the file in ReadOnly mode
try:
    f = open(filename, 'r') #iterable
except IOError:
    sys.exit("Could not open input file!")

# Used to show the progress
print "Beginning to send chunks"
sys.stdout.flush()

# While the generator returns new chunk, sent them to the engines
for new_chunk in get_chunk(f,lines_per_block):
    
    #After the first loop, first_chunk is False. 
    first_chunk = False
    
    #Decide the engine to be used to classify the new chunk
    run_engine = c_engines.next()
    
    # Wait until the engine is ready
    while ( not async_tasks[run_engine].ready() ):
        time.sleep(1)
    
    #Send data to the assigned engine.
    mydict = dict(data = new_chunk)
    
    # The data is sent to the engine in blocking mode. The push function does not return
    # until the engine has received the data. 
    engines[run_engine].push(mydict,block=True)

    # We execute the classification task on the engine. Observe that the task is executed
    # in non-blocking mode. Thus the execute function reurns immediately. 
    async_tasks[run_engine] = engines[run_engine].execute('process(data)', block=False)
    
    # Increase the counter    
    chunk_n += 1

    # Update the progress
    if chunk_n % 1000 == 0:
        print "Chunks sent until this moment: " + str(chunk_n)
        sys.stdout.flush()

print "All chunks have been sent"
sys.stdout.flush()
# Get the results from each engine and accumulate in global_result
for engine in engines.ids:
    # Be sure that all async tasks are finished
    while ( not async_tasks[engine].ready() ):
        time.sleep(1)
    global_result += engines[engine].pull('local_total', block=True)

#Close the file
f.close()

print "Total number of chunks processed: " + str(chunk_n)
print "---------------------------------------------"
print "Agregated dictionary"
print "---------------------------------------------"
print dict(global_result)

print "Time in seconds: ", time.time() - t0
sys.stdout.flush()

