# Import modules
import pandas as pd

raw_data = {'first_name': ['Jason', 'Jason', 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Miller', 'Ali', 'Milner', 'Cooze'], 
        'age': [42, 42, 36, 24, 73], 
        'preTestScore': [4, 4, 31, 2, 3],
        'postTestScore': [25, 25, 57, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
df

# View preTestscore where postTestscore is greater than 50
df['preTestScore'].where(df['postTestScore'] > 50)

