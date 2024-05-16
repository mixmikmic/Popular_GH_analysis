commander_names = ["Alan Brooke", "George Marshall", "Frank Jack Fletcher", "Conrad Helfrich", "Albert Kesselring"] 

# Sort a variable called 'commander_names' by the last elements of each name.
sorted(commander_names, key=lambda x: x.split(" ")[-1])

