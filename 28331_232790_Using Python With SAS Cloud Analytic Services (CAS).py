import swat

conn = swat.CAS('cas01', 49786)

conn.serverstatus()

conn.userinfo()

conn.help();

tbl2 = conn.read_csv('https://raw.githubusercontent.com/'
                    'sassoftware/sas-viya-programming/master/data/cars.csv', 
                     casout=conn.CASTable('cars'))
tbl2

conn.tableinfo()

tbl = conn.CASTable('attrition')

tbl.columninfo()

get_ipython().magic('pinfo tbl2')

tbl2.fetch()

tbl.summary() 

tbl.freq(inputs='Attrition')

conn.loadactionset('regression')
conn.help(actionset='regression');

output = tbl.logistic(
    target='Attrition',
    inputs=['Gender', 'MaritalStatus', 'AccountAge'],
    nominals = ['Gender', 'MaritalStatus']
) 

output.keys()

output

from swat.render import render_html
render_html(output)

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/'
                 'sassoftware/sas-viya-programming/master/data/cars.csv')
df.describe()

tbl2.describe()

tbl2.groupby('Origin').describe()

tbl[['Gender', 'AccountAge']].head()

from bokeh.plotting import show, figure
from bokeh.charts import Bar
from bokeh.io import output_notebook
output_notebook()

output1 = tbl.freq(inputs=['Attrition'])

p = Bar(output1['Frequency'], 'FmtVar', 
        values='Frequency',
        color="#1f77b4", 
        agg='mean', 
        title="", 
        xlabel = "Attrition",
        ylabel = 'Frequency',        
        bar_width=0.8,
        plot_width=600, 
        plot_height=400 
)
show(p)

conn.tableinfo()

tbl2.groupby(['Origin', 'Type']).describe()

tbl2[['MPG_CITY', 'MPG_Highway', 'MSRP']].describe()

tbl2[(tbl2.MSRP > 90000) & (tbl2.Cylinders < 12)].head()

conn.runcode(code='''
    data cars_temp;
        set cars;
        sqrt_MSRP = sqrt(MSRP);
        MPG_avg = (MPG_city + MPG_highway) / 2;
    run;
''')

conn.tableinfo()

conn.loadactionset('fedsql')

conn.fedsql.execdirect(query='''
    select make, model, msrp,
    mpg_highway from cars
        where msrp > 80000 and mpg_highway > 20
''')

