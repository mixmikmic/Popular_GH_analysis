# Import the required modules
from __future__ import division
import numpy as np
import csv

file_name = "../data/ranking-algorithm/epl_16_17.csv"
x = [];
# Read the CSV files
with open(file_name, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader)
    for row in csvreader:
        x.append([row[2], row[3], int(row[4]), int(row[5])])

# The team names
teams = {x: i for i, x in enumerate({match[0] for match in x})}
teams_rev = {v: k for k, v in teams.items()} 
# Convert into nice numpy array
x = np.array([[teams[match[0]], teams[match[1]], match[2], match[3]] for match in x], dtype=np.int32)
no_teams = len(teams)

# Prepare the Transition matrix
trans = np.zeros((no_teams, no_teams), dtype=np.float32)
for match_id in xrange(x.shape[0]):
    i, j = x[match_id][0], x[match_id][1]
    i_score, j_score = x[match_id][2], x[match_id][3]
    den = i_score + j_score
    if den > 0:
        trans[i, i] += (i_score > j_score) +  i_score / den
        trans[j, j] += (i_score < j_score) +  j_score / den
        trans[i, j] += (i_score < j_score) +  j_score / den
        trans[j, i] += (i_score > j_score) +  i_score / den

# Normalize the transition matrix
norm = np.sum(trans, axis=1) 
trans_norm = trans / np.expand_dims(norm, axis=0)

# Perform the eigenvalue decomposition of the transition matrix
w, v = np.linalg.eig(trans_norm.T)
# Normalize the 1st eigenvector that corresponds to eigenvalue = 1
s_d = v[:, 0].real / np.sum(v[:, 0].real)
# Sort s_d
sorted_ranking = np.argsort(s_d)[::-1]
# Prepare a list to display 
best_teams = [(teams_rev[i], s_d[i]) for i in sorted_ranking]

print("The rankings of the teams are")
for team in best_teams:
    print team

