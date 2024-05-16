import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

mnist_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/mnist.csv.gz', sep='\t', compression='gzip')
mnist_data.head()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(8, 8))

for record_num in range(1, 65):
    plt.subplot(8, 8, record_num)
    
    digit_features = mnist_data.iloc[record_num].drop('class').values
    sb.heatmap(digit_features.reshape((28, 28)),
               cmap='Greys',
               square=True, cbar=False,
               xticklabels=[], yticklabels=[])

plt.tight_layout()
("")

cv_scores = cross_val_score(RandomForestClassifier(n_estimators=10, n_jobs=-1),
                            X=mnist_data.drop('class', axis=1).values,
                            y=mnist_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))

cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            X=mnist_data.drop('class', axis=1).values,
                            y=mnist_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

hill_valley_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/Hill_Valley_without_noise.csv.gz', sep='\t', compression='gzip')
hill_valley_data.head()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

with plt.style.context('seaborn-notebook'):
    plt.figure(figsize=(6, 6))
    for record_num in range(1, 11):
        plt.subplot(10, 1, record_num)
        hv_record_features = hill_valley_data.loc[record_num].drop('class').values
        plt.plot(hv_record_features)
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
("")

cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            X=hill_valley_data.drop('class', axis=1).values,
                            y=hill_valley_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))

cv_scores = cross_val_score(LogisticRegression(),
                            X=hill_valley_data.drop('class', axis=1).values,
                            y=hill_valley_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score

hill_valley_noisy_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/Hill_Valley_with_noise.csv.gz', sep='\t', compression='gzip')
hill_valley_noisy_data.head()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

with plt.style.context('seaborn-notebook'):
    plt.figure(figsize=(6, 6))
    for record_num in range(1, 11):
        plt.subplot(10, 1, record_num)
        hv_noisy_record_features = hill_valley_noisy_data.loc[record_num].drop('class').values
        plt.plot(hv_noisy_record_features)
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
("")

cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            X=hill_valley_noisy_data.drop('class', axis=1).values,
                            y=hill_valley_noisy_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))

cv_scores = cross_val_score(make_pipeline(PCA(n_components=10), RandomForestClassifier(n_estimators=100, n_jobs=-1)),
                            X=hill_valley_noisy_data.drop('class', axis=1).values,
                            y=hill_valley_noisy_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))

import pandas as pd
from sklearn.cross_validation import train_test_split
from tpot import TPOTClassifier

hill_valley_noisy_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/Hill_Valley_with_noise.csv.gz', sep='\t', compression='gzip')
hill_valley_noisy_data.head()

X = hill_valley_noisy_data.drop('class', axis=1).values
y = hill_valley_noisy_data.loc[:, 'class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

my_tpot = TPOTClassifier(generations=10, verbosity=2)
my_tpot.fit(X_train, y_train)
print(my_tpot.score(X_test, y_test))

