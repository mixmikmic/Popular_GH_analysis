get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd
data = pd.read_csv('estimating_lead.csv')

# Run this to see a printout of the data
data

# The object 'data' is a pandas DataFrame
# Don't worry if you don't know what that is, we can turn it into a numpy array
datamatrix = data.as_matrix()

# We can use some pandas magic to do these counts quickly

N = data.shape[0]
params = {
    'pi_mle': data[data.Lead == 'Lead'].shape[0] / N,
    'theta_haslead_isold': data[(data.Lead == 'Lead') & (data.IsOld == True) ].shape[0] / \
                           data[data.Lead == 'Lead'].shape[0],
    'theta_nothaslead_isold': data[(data.Lead != 'Lead') & (data.IsOld == True) ].shape[0] / \
                           data[data.Lead != 'Lead'].shape[0],
    'theta_haslead_rsl': data[(data.Lead == 'Lead') & (data.RecordSaysLead == True) ].shape[0] / \
                           data[data.Lead == 'Lead'].shape[0],
    'theta_nothaslead_rsl': data[(data.Lead != 'Lead') & (data.RecordSaysLead == True) ].shape[0] / \
                           data[data.Lead != 'Lead'].shape[0],
}
print(pd.Series(params))

