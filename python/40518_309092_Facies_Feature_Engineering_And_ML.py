get_ipython().magic('matplotlib notebook')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm

filename = 'training_data.csv'
training_data = pd.read_csv(filename)

## Create a difference vector for each feature e.g. x1-x2, x1-x3... x2-x3...

# order features in depth.

feature_vectors = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
feature_vectors = feature_vectors[['Depth','GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]

def difference_vector(feature_vectors):
    length = len(feature_vectors['Depth'])
    df_temp = np.zeros((25, length))
                          
    for i in range(0,int(len(feature_vectors['Depth']))):
                       
        vector_i = feature_vectors.iloc[i,:]
        vector_i = vector_i[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
        for j, value_j in enumerate(vector_i):
            for k, value_k in enumerate(vector_i): 
                differ_j_k = value_j - value_k          
                df_temp[5*j+k, i] = np.abs(differ_j_k)
                
    return df_temp

def diff_vec2frame(feature_vectors, df_temp):
    
    heads = feature_vectors.columns[1::]
    for i in range(0,5):
        string_i = heads[i]
        for j in range(0,5):
            string_j = heads[j]
            col_head = 'diff'+string_i+string_j
            
            df = pd.Series(df_temp[5*i+j, :])
            feature_vectors[col_head] = df
    return feature_vectors
            
df_diff = difference_vector(feature_vectors)    
feature_vectors = diff_vec2frame(feature_vectors, df_diff)

# Drop duplicated columns and column of zeros
feature_vectors = feature_vectors.T.drop_duplicates().T   
feature_vectors.drop('diffGRGR', axis = 1, inplace = True)

# Add Facies column back into features vector

feature_vectors['Facies'] = training_data['Facies']

# # group by facies, take statistics of each facies e.g. mean, std. Take sample difference of each row with 

def facies_stats(feature_vectors):
    facies_labels = np.sort(feature_vectors['Facies'].unique())
    frame_mean = pd.DataFrame()
    frame_std = pd.DataFrame()
    for i, value in enumerate(facies_labels):
        facies_subframe = feature_vectors[feature_vectors['Facies']==value]
        subframe_mean = facies_subframe.mean()
        subframe_std = facies_subframe.std()
        
        frame_mean[str(value)] = subframe_mean
        frame_std[str(value)] = subframe_std
    
    return frame_mean.T, frame_std.T

def feature_stat_diff(feature_vectors, frame_mean, frame_std):
    
    feature_vec_origin = feature_vectors[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
    
    for i, column in enumerate(feature_vec_origin):
        
        feature_column = feature_vec_origin[column]
        stat_column_mean = frame_mean[column]
        stat_column_std = frame_std[column]
        
        for j in range(0,9):
            
            stat_column_mean_facie = stat_column_mean[j]
            stat_column_std_facie = stat_column_std[j]
            
            feature_vectors[column + '_mean_diff_facies' + str(j)] = feature_column-stat_column_mean_facie
            feature_vectors[column + '_std_diff_facies' + str(j)] = feature_column-stat_column_std_facie
    return feature_vectors
             
frame_mean, frame_std = facies_stats(feature_vectors)  
feature_vectors = feature_stat_diff(feature_vectors, frame_mean, frame_std)

# A = feature_vectors.sort_values(by='Facies')
# A.reset_index(drop=True).plot(subplots=True, style='b', figsize = [12, 400])

df = feature_vectors
predictors = feature_vectors.columns
predictors = list(predictors.drop('Facies'))
correct_facies_labels = df['Facies'].values
# Scale features
df = df[predictors]

scaler = preprocessing.StandardScaler().fit(df)
scaled_features = scaler.transform(df)

# Train test split:

X_train, X_test, y_train, y_test = train_test_split(scaled_features,  correct_facies_labels, test_size=0.2, random_state=0)
alg = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=8, min_samples_leaf=3, max_features= None)
alg.fit(X_train, y_train)

predicted_random_forest = alg.predict(X_test)

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
result = predicted_random_forest
conf = confusion_matrix(y_test, result)
display_cm(conf, facies_labels, hide_zeros=True, display_metrics = True)

def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

print(accuracy(conf))

adjacent_facies = np.array([[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]])

def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))

print(accuracy_adjacent(conf, adjacent_facies))

# read in Test data

filename = 'validation_data_nofacies.csv'
test_data = pd.read_csv(filename)

# Reproduce feature generation

feature_vectors_test = test_data.drop(['Formation', 'Well Name'], axis=1)
feature_vectors_test = feature_vectors_test[['Depth','GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]

df_diff_test = difference_vector(feature_vectors_test)    
feature_vectors_test = diff_vec2frame(feature_vectors_test, df_diff_test)

# Drop duplicated columns and column of zeros

feature_vectors_test = feature_vectors_test.T.drop_duplicates().T   
feature_vectors_test.drop('diffGRGR', axis = 1, inplace = True)

# Create statistical feature differences using preivously caluclated mean and std values from train data.

feature_vectors_test = feature_stat_diff(feature_vectors_test, frame_mean, frame_std)
feature_vectors_test = feature_vectors_test[predictors]
scaler = preprocessing.StandardScaler().fit(feature_vectors_test)
scaled_features = scaler.transform(feature_vectors_test)

predicted_random_forest = alg.predict(scaled_features)


predicted_random_forest
test_data['Facies'] = predicted_random_forest
test_data.to_csv('test_data_prediction_CE.csv')



