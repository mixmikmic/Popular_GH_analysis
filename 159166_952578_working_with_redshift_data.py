get_ipython().system('conda install -y -c anaconda psycopg2')

import os
import boto3
import pandas as pd
import json
import psycopg2
import sqlalchemy as sa

region = boto3.Session().region_name

bucket='<your_s3_bucket_name_here>' # put your s3 bucket name here, and create s3 bucket
prefix = 'sagemaker/redshift'
# customize to your bucket where you have stored the data

credfile = 'redshift_creds_template.json.nogit'

# Read credentials to a dictionary
with open(credfile) as fh:
    creds = json.loads(fh.read())

# Sample query for testing
query = 'select * from public.irisdata;'

print("Reading from Redshift...")

def get_conn(creds): 
    conn = psycopg2.connect(dbname=creds['db_name'], 
                            user=creds['username'], 
                            password=creds['password'],
                            port=creds['port_num'],
                            host=creds['host_name'])
    return conn

def get_df(creds, query):
    with get_conn(creds) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result_set = cur.fetchall()
            colnames = [desc.name for desc in cur.description]
            df = pd.DataFrame.from_records(result_set, columns=colnames)
    return df

df = get_df(creds, query)

print("Saving file")
localFile = 'iris.csv'
df.to_csv(localFile, index=False)

print("Done")

print("Writing to S3...")

fObj = open(localFile, 'rb')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, localFile)).upload_fileobj(fObj)
print("Done")

print("Reading from S3...")
# key unchanged for demo purposes - change key to read from output data
key = os.path.join(prefix, localFile)

s3 = boto3.resource('s3')
outfile = 'iris2.csv'
s3.Bucket(bucket).download_file(key, outfile)
df2 = pd.read_csv(outfile)
print("Done")

print("Writing to Redshift...")

connection_str = 'postgresql+psycopg2://' +                   creds['username'] + ':' +                   creds['password'] + '@' +                   creds['host_name'] + ':' +                   creds['port_num'] + '/' +                   creds['db_name'];
                    
df2.to_sql('irisdata_v2', connection_str, schema='public', index=False)
print("Done")

pd.options.display.max_rows = 2
conn = get_conn(creds)
query = 'select * from irisdata3'
df = pd.read_sql_query(query, conn)
df

