import pandas as pd
from altair import Chart, X, Y, Axis, SortField

get_ipython().magic('matplotlib inline')

budget = pd.read_csv("https://github.com/chris1610/pbpython/raw/master/data/mn-budget-detail-2014.csv")

budget.head()

budget_top_10 = budget.sort_values(by='amount',ascending=False)[:10]

budget_top_10.plot(kind="bar",x=budget_top_10["detail"],
                   title="MN Capital Budget - 2014",
                   legend=False)

c = Chart(budget_top_10).mark_bar().encode(
    x='detail',
    y='amount')
c

c = Chart(budget_top_10).mark_bar().encode(
    y='detail',
    x='amount')
c

c.to_dict(data=False)

Chart(budget_top_10).mark_bar().encode(
    x=X('detail'),
    y=Y('amount')
)

Chart(budget_top_10).mark_bar().encode(
    x=X('detail'),
    y=Y('amount'),
    color="category"
)

Chart(budget).mark_bar().encode(
    x='detail',
    y='amount',
    color='category')

Chart(budget).mark_bar().encode(
    x='detail:N',
    y='amount:Q',
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )

Chart(budget).mark_bar().encode(
    y='detail:N',
    x='amount:Q',
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )

Chart(budget).mark_bar().encode(
    x=X('detail:O',
        axis=Axis(title='Project')),
    y=Y('amount:Q',
        axis=Axis(title='2014 Budget')),
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )

Chart(budget).mark_bar().encode(
    x=X('detail:O', sort=SortField(field='amount', order='descending', op='sum'),
        axis=Axis(title='Project')),
    y=Y('amount:Q',
        axis=Axis(title='2014 Budget')),
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )

Chart(budget).mark_bar().encode(
    x=X('category', sort=SortField(field='amount', order='descending', op='sum'),
        axis=Axis(title='Category')),
    y=Y('sum(amount)',
        axis=Axis(title='2014 Budget')))

c = Chart(budget).mark_bar().encode(
    y=Y('category', sort=SortField(field='amount', order='descending', op='sum'),
        axis=Axis(title='Category')),
    x=X('sum(amount)',
        axis=Axis(title='2014 Budget')))
c

c.to_dict(data=False)



