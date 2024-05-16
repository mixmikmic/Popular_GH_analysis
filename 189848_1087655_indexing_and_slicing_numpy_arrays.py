# Import Modules
import numpy as np

# Create an array of battle casualties from the first to the last battle
battleDeaths = np.array([1245, 2732, 3853, 4824, 5292, 6184, 7282, 81393, 932, 10834])

# Divide the array of battle deaths into start, middle, and end of the war
warStart = battleDeaths[0:3]; print('Death from battles at the start of war:', warStart)
warMiddle = battleDeaths[3:7]; print('Death from battles at the middle of war:', warMiddle)
warEnd = battleDeaths[7:10]; print('Death from battles at the end of war:', warEnd)

# Change the battle death numbers from the first battle
warStart[0] = 11101

# View that change reflected in the warStart slice of the battleDeaths array
warStart

# View that change reflected in (i.e. "broadcasted to) the original battleDeaths array
battleDeaths

# Create an array of regiment information
regimentNames = ['Nighthawks', 'Sky Warriors', 'Rough Riders', 'New Birds']
regimentNumber = [1, 2, 3, 4]
regimentSize = [1092, 2039, 3011, 4099]
regimentCommander = ['Mitchell', 'Blackthorn', 'Baker', 'Miller']

regiments = np.array([regimentNames, regimentNumber, regimentSize, regimentCommander])
regiments

# View the first column of the matrix
regiments[:,0]

# View the second row of the matrix
regiments[1,]

# View the top-right quarter of the matrix
regiments[:2,2:]

