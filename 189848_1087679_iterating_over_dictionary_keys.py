Officers = {'Michael Mulligan': 'Red Army',
            'Steven Johnson': 'Blue Army',
            'Jessica Billars': 'Green Army',
            'Sodoni Dogla': 'Purple Army',
            'Chris Jefferson': 'Orange Army'}

Officers

# Display all dictionary entries where the key doesn't start with 'Chris'
{keys : Officers[keys] for keys in Officers if not keys.startswith('Chris')}

