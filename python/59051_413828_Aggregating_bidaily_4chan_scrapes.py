import boto
import boto3
import pandas as pd
import re
from IPython.display import clear_output

session = boto3.Session(profile_name='default')
s3 = session.resource('s3')
bucket = s3.Bucket("far-right")
session.available_profiles

base_url = 's3:far-right/'
match_string = "info-source/daily/[0-9]+/fourchan/fourchan"

files = []
print("Getting bucket and files info")
for obj in bucket.objects.all():
    if bool(re.search(match_string, obj.key)):
        files.append(obj.key)
        
df = pd.DataFrame()
for i, file in enumerate(files):
    clear_output()
    print("Loading file: " + str(i + 1) + " out of " + str(len(files)))
    if df.empty:
        df = pd.read_json(base_url + file)        
    else:
        df = pd.concat([df, pd.read_json(base_url + file)])
    
clear_output()
print("Completed Loading Files")

df.shape

df.head()



