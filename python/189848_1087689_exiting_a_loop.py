# Create a list:
armies = ['Red Army', 'Blue Army', 'Green Army']

for army in armies:
    print(army)
    if army == 'Blue Army':
        print('Blue Army Found! Stopping.')
        break

for army in armies:
    print(army)
    if army == 'Orange Army':
        break
else:
    print('Looped Through The Whole List, No Orange Army Found')

