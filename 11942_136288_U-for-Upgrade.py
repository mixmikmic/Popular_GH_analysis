# Define the version number of this VirtualBox Appliance - dates will sort 'alphabetically'
with open('../config/vbox_name', 'rt') as f:
    repo,dt,tm=f.read().strip().split('_')
dt_tm = "%s_%s" % (dt,tm)
repo,dt,tm

# Repo to download from :
repo_base='https://raw.githubusercontent.com/mdda/deep-learning-workshop/'
path_to_root='master'
root_to_updates='/notebooks/model/updates.py'

#  Download the changes 'script', so we can find the changes that were made after this VM was created
import requests

updates = requests.get(repo_base+path_to_root+root_to_updates)
if updates.status_code == 200:
    with open('model/updates_current.py', 'wb') as f:
        f.write(updates.content)
        print("file : updates_current.py downloaded successfully")
else:
    print("Download unsuccessful : Complain!")    

get_ipython().magic('load model/updates_current.py')

