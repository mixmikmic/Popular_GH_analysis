import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

# Create an empty dataset
df = pd.DataFrame()

# Create two variables called x0 and x1. Make the first value of x1 a missing value
df['x0'] = [0.3051,0.4949,0.6974,np.nan,0.2231,np.nan,0.4436,0.5897,0.6308,0.5]
df['x1'] = [np.nan,0.2654,0.2615,0.5846,0.4615,np.nan,0.4962,0.3269,np.nan,0.6731]

# View the dataset
df

# Create an imputer object that looks for 'Nan' values
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(df)

# Apply the imputer to the df dataset
imputed_df = mean_imputer.transform(df.values)





