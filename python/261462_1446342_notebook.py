# Nothing to do here

# Import modules
import pandas as pd

# Read colors data
colors = pd.read_csv('datasets/colors.csv')

# Print the first few rows
colors.head()

# How many distinct colors are available?
num_colors = colors.shape[0]
print(num_colors)

# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby(colors['is_trans']).count()
colors_summary

get_ipython().run_line_magic('matplotlib', 'inline')
# Read sets data as `sets`
sets = pd.read_csv('datasets/sets.csv')
# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets[['year', 'num_parts']].groupby('year', as_index=False).count()
# Plot trends in average number of parts by year
parts_by_year.plot(x = 'year', y = 'num_parts')
parts_by_year.head()

# themes_by_year: Number of themes shipped by year
themes_by_year = sets[['year', 'theme_id']].groupby('year', as_index=False).count()
themes_by_year.head()

# Nothing to do here

