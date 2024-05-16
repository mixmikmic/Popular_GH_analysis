import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_excel('data/Governance.xlsx', sheetname=0)
melt_data = data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name="Year")
melt_data = melt_data[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'value']]

melt_data.head(5)

copy_melt = melt_data.copy()

copy_melt = pd.pivot_table(copy_melt, values = 'value', index=['Country Name', 'Country Code','Year'], columns = 'Indicator Name').reset_index()
#copy_melt.index = piv_data['Year']
copy_melt.head(5)

#copy_melt.index = copy_melt.Year

copy_melt.head()

control_corruption = ['Year','Country Name','Control of Corruption: Estimate', 'Control of Corruption: Number of Sources',
                      'Control of Corruption: Percentile Rank', 
                      'Control of Corruption: Percentile Rank, Lower Bound of 90% Confidence Interval',
                      'Control of Corruption: Percentile Rank, Upper Bound of 90% Confidence Interval',
                      'Control of Corruption: Standard Error']
control_data = copy_melt[control_corruption]

government_effectivesness = ['Year','Country Name','Government Effectiveness: Estimate', 'Government Effectiveness: Number of Sources',
                             'Government Effectiveness: Percentile Rank', 
                             'Government Effectiveness: Percentile Rank, Lower Bound of 90% Confidence Interval',
                             'Government Effectiveness: Percentile Rank, Upper Bound of 90% Confidence Interval',
                             'Government Effectiveness: Standard Error']
government_data = copy_melt[government_effectivesness]
government_data.head(5)

political_stab = ['Year','Country Name','Political Stability and Absence of Violence/Terrorism: Estimate',
                  'Political Stability and Absence of Violence/Terrorism: Number of Sources',
                  'Political Stability and Absence of Violence/Terrorism: Percentile Rank',
                  'Political Stability and Absence of Violence/Terrorism: Percentile Rank, Lower Bound of 90% Confidence Interval',
                  'Political Stability and Absence of Violence/Terrorism: Percentile Rank, Upper Bound of 90% Confidence Interval',
                  'Political Stability and Absence of Violence/Terrorism: Standard Error']
political_data = copy_melt[political_stab]

regulatory_quality = ['Year','Country Name','Regulatory Quality: Estimate', 'Regulatory Quality: Number of Sources',
                      'Regulatory Quality: Percentile Rank', 
                      'Regulatory Quality: Percentile Rank, Lower Bound of 90% Confidence Interval',
                      'Regulatory Quality: Percentile Rank, Upper Bound of 90% Confidence Interval',
                      'Regulatory Quality: Standard Error']
regulatory_data = copy_melt[regulatory_quality]

rule_law = ['Year','Country Name','Rule of Law: Estimate', 'Rule of Law: Number of Sources', 'Rule of Law: Percentile Rank',
            'Rule of Law: Percentile Rank, Lower Bound of 90% Confidence Interval',
            'Rule of Law: Percentile Rank, Upper Bound of 90% Confidence Interval',
            'Rule of Law: Standard Error']
rule_data = copy_melt[rule_law]

voice_and_account = ['Year','Country Name','Voice and Accountability: Estimate', 'Voice and Accountability: Number of Sources',
                     'Voice and Accountability: Percentile Rank',
                     'Voice and Accountability: Percentile Rank, Lower Bound of 90% Confidence Interval',
                     'Voice and Accountability: Percentile Rank, Upper Bound of 90% Confidence Interval',
                     'Voice and Accountability: Standard Error']
voice_data = copy_melt[voice_and_account]

voice_data.to_csv('data/voice_accountability.csv',encoding='utf-8', index=False)
rule_data.to_csv('data/rule_of_law.csv',encoding='utf-8', index=False)
regulatory_data.to_csv('data/regulatory_quality.csv',encoding='utf-8', index=False)
political_data.to_csv('data/political_stability.csv',encoding='utf-8', index=False)
control_data.to_csv('data/control_corruption.csv',encoding='utf-8', index=False)
government_data.to_csv('data/government_effectiveness.csv',encoding='utf-8', index=False)

voice_data.head()

vvv = pd.read_csv('data/voice_accountability.csv')
vvv.head()





voice_data.groupby(lambda x: pd.to_datetime(x))
voice_data.sort_values('Year').head()

SSA = ["Angola", "Gabon", "Nigeria", "Benin", "Gambia, The", "Rwanda", "Guinea-Bissau","Botswana", 
       "Ghana", "São Tomé and Principe", "Burkina Faso", "Guinea", "Senegal", "Burundi", "Seychelles", 
       "Cabo Verde", "Kenya", "Sierra Leone", "Cameroon", "Lesotho", "Somalia", "Central African Republic", 
       "Liberia", "South Africa", "Chad", "Madagascar", "Comoros", "Malawi", "Sudan", "Congo, Dem. Rep.", 
       "Mali", "Swaziland", "Congo, Rep", "Mauritania", "Tanzania", "Côte d'Ivoire", "Mauritius", "Togo", 
       "Equatorial Guinea", "Mozambique", "Uganda", "Eritrea" "Namibia", "Zambia", "Ethiopia", "Niger", "Zimbabwe"]
ssa_melt = voice_data[voice_data['Country Name'].isin(SSA)]

ssa_melt['Country Name'].nunique()

ssa_melt.head()

est_voice = ssa_melt[ssa_melt['Country Name'] == 'Somalia'].groupby('Voice and Accountability: Estimate').size().head(10).to_frame(name = 'count').reset_index()

ssa_melt.groupby('Country Name')['Voice and Accountability: Number of Sources'].mean()





ax = ssa_melt.plot(x='Year', y=["Country Name","Voice and Accountability: Number of Sources"])













table = pivot_table(df, values='D', index=['Somalia'], columns=['C'], aggfunc=np.sum)

get_ipython().run_line_magic('pinfo', 'pd.pivot_table')

estimate = voice_data.groupby('Country Name')['Voice and Accountability: Estimate'].sum()

np.argsort(estimate)





