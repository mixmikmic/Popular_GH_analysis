import pandas as pd
from pprint import pprint
from github import Github

# Load Github token 
github_token = pd.read_csv("../github_token.csv")
ACCESS_TOKEN = github_token.values.flatten()[0]

# Set user and repo of interest
USER = 'apache'
REPO = 'spark'

githubx = Github(ACCESS_TOKEN, per_page=100)
user = githubx.get_user(USER)
repo = user.get_repo(REPO)


# Retrieve key facts from the user - Apache. 
# repos_apache = [repo.name for repo in githubx.get_user('apache').get_repos()]

# Retrieve key facts from the user - Apache.
# apache_repos = [repo.name for repo in repos]
repos_apache = [rp.name for rp in githubx.get_user('apache').get_repos()]
print("\n User '{}' has {} repos \n".format(USER, len(apache_repos)))

# Check if project Spark exists
'spark' in apache_repos

print("Programming Languages used under the `Spark` project are: \n")
pp(rp.get_languages())

stargazers = [s for s in repo.get_stargazers()]
print("Number of Stargazers is {}".format(len(stargazers)))

# Retrieve a few key participants of the wide Spark GitHub repository network.
# The first stargazer is Matei Zaharia, the cofounder of the Spark project when he was doing his PhD in Berkeley.

[stargazers[i].login for i in range(0,20)]



