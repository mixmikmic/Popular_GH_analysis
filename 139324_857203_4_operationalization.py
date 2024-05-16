## setup our environment by importing required libraries
import json
import os
import shutil
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

# setup the pyspark environment
from pyspark.sql import SparkSession

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

# For Azure blob storage access
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess

# For logging model evaluation parameters back into the
# AML Workbench run history plots.
import logging
from azureml.logging import get_azureml_logger

amllog = logging.getLogger("azureml")
amllog.level = logging.INFO

# Turn on cell level logging.
get_ipython().run_line_magic('azureml', 'history on')
get_ipython().run_line_magic('azureml', 'history show')

# Time the notebook execution. 
# This will only make sense if you "Run all cells"
tic = time.time()

logger = get_azureml_logger() # logger writes to AMLWorkbench runtime view
spark = SparkSession.builder.getOrCreate()

# Telemetry
logger.log('amlrealworld.predictivemaintenance.operationalization','true')

# Enter your Azure blob storage details here 
ACCOUNT_NAME = "<your blob storage account name>"

# You can find the account key under the _Access Keys_ link in the 
# [Azure Portal](portal.azure.com) page for your Azure storage container.
ACCOUNT_KEY = "<your blob storage account key>"
#-------------------------------------------------------------------------------------------
# We will create this container to hold the results of executing this notebook.
# If this container name already exists, we will use that instead, however
# This notebook will ERASE ALL CONTENTS.
CONTAINER_NAME = "featureengineering"
FE_DIRECTORY = 'featureengineering_files.parquet'

MODEL_CONTAINER = 'modeldeploy'

# Connect to your blob service     
az_blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

# Create a new container if necessary, otherwise you can use an existing container.
# This command creates the container if it does not already exist. Else it does nothing.
az_blob_service.create_container(CONTAINER_NAME, 
                                 fail_on_exist=False, 
                                 public_access=PublicAccess.Container)

# create a local path where to store the results later.
if not os.path.exists(FE_DIRECTORY):
    os.makedirs(FE_DIRECTORY)

# download the entire parquet result folder to local path for a new run 
for blob in az_blob_service.list_blobs(CONTAINER_NAME):
    if CONTAINER_NAME in blob.name:
        local_file = os.path.join(FE_DIRECTORY, os.path.basename(blob.name))
        az_blob_service.get_blob_to_path(CONTAINER_NAME, blob.name, local_file)

fedata = spark.read.parquet(FE_DIRECTORY)

fedata.limit(5).toPandas().head(5)

def init():
    # read in the model file
    from pyspark.ml import PipelineModel
    global pipeline
    
    pipeline = PipelineModel.load(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']+'pdmrfull.model')
    
def run(input_df):
    import json
    response = ''
    try:
        #Get prediction results for the dataframe
        
        # We'll use the known label, key variables and 
        # a few extra columns we won't need.
        key_cols =['label_e','machineID','dt_truncated', 'failure','model_encoded','model' ]

        # Then get the remaing feature names from the data
        input_features = input_df.columns

        # Remove the extra stuff if it's in the input_df
        input_features = [x for x in input_features if x not in set(key_cols)]
        
        # Vectorize as in model building
        va = VectorAssembler(inputCols=(input_features), outputCol='features')
        data = va.transform(input_df).select('machineID','features')
        score = pipeline.transform(data)
        predictions = score.collect()

        #Get each scored result
        preds = [str(x['prediction']) for x in predictions]
        response = ",".join(preds)
    except Exception as e:
        print("Error: {0}",str(e))
        return (str(e))
    
    # Return results
    print(json.dumps(response))
    return json.dumps(response)

# define the input data frame
inputs = {"input_df": SampleDefinition(DataTypes.SPARK, 
                                       fedata.drop("dt_truncated","failure","label_e", "model","model_encoded"))}

json_schema = generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')

# We'll use the known label, key variables and 
# a few extra columns we won't need. (machineID is required)
key_cols =['label_e','dt_truncated', 'failure','model_encoded','model' ]

# Then get the remaining feature names from the data
input_features = fedata.columns
# Remove the extra stuff if it's in the input_df
input_features = [x for x in input_features if x not in set(key_cols)]


# this is an example input data record
input_data = [[114, 163.375732902,333.149484586,100.183951698,44.0958812638,164.114723991,
               277.191815232,97.6289110707,50.8853505161,21.0049565219,67.5287259378,12.9361526861,
               4.61359760918,15.5377738062,67.6519885441,10.528274633,6.94129487555,0.0,0.0,0.0,
               0.0,0.0,489.0,549.0,549.0,564.0,18.0]]

df = (spark.createDataFrame(input_data, input_features))

# test init() in local notebook
init()

# test run() in local notebook
run(df)

# save the schema file for deployment
out = json.dumps(json_schema)
with open(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + 'service_schema.json', 'w') as f:
    f.write(out)

get_ipython().run_cell_magic('writefile', "{os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']}/pdmscore.py", '\nimport json\nfrom pyspark.ml import Pipeline\nfrom pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier\n\n# for creating pipelines and model\nfrom pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer\n\ndef init():\n    # read in the model file\n    from pyspark.ml import PipelineModel\n    # read in the model file\n    global pipeline\n    pipeline = PipelineModel.load(\'pdmrfull.model\')\n    \ndef run(input_df):\n    response = \'\'\n    try:\n       \n        # We\'ll use the known label, key variables and \n        # a few extra columns we won\'t need.\n        key_cols =[\'label_e\',\'machineID\',\'dt_truncated\', \'failure\',\'model_encoded\',\'model\' ]\n\n        # Then get the remaing feature names from the data\n        input_features = input_df.columns\n\n        # Remove the extra stuff if it\'s in the input_df\n        input_features = [x for x in input_features if x not in set(key_cols)]\n        \n        # Vectorize as in model building\n        va = VectorAssembler(inputCols=(input_features), outputCol=\'features\')\n        data = va.transform(input_df).select(\'machineID\',\'features\')\n        score = pipeline.transform(data)\n        predictions = score.collect()\n\n        #Get each scored result\n        preds = [str(x[\'prediction\']) for x in predictions]\n        response = ",".join(preds)\n    except Exception as e:\n        print("Error: {0}",str(e))\n        return (str(e))\n    \n    # Return results\n    print(json.dumps(response))\n    return json.dumps(response)\n\nif __name__ == "__main__":\n    init()\n    run("{\\"input_df\\":[{\\"machineID\\":114,\\"volt_rollingmean_3\\":163.375732902,\\"rotate_rollingmean_3\\":333.149484586,\\"pressure_rollingmean_3\\":100.183951698,\\"vibration_rollingmean_3\\":44.0958812638,\\"volt_rollingmean_24\\":164.114723991,\\"rotate_rollingmean_24\\":277.191815232,\\"pressure_rollingmean_24\\":97.6289110707,\\"vibration_rollingmean_24\\":50.8853505161,\\"volt_rollingstd_3\\":21.0049565219,\\"rotate_rollingstd_3\\":67.5287259378,\\"pressure_rollingstd_3\\":12.9361526861,\\"vibration_rollingstd_3\\":4.61359760918,\\"volt_rollingstd_24\\":15.5377738062,\\"rotate_rollingstd_24\\":67.6519885441,\\"pressure_rollingstd_24\\":10.528274633,\\"vibration_rollingstd_24\\":6.94129487555,\\"error1sum_rollingmean_24\\":0.0,\\"error2sum_rollingmean_24\\":0.0,\\"error3sum_rollingmean_24\\":0.0,\\"error4sum_rollingmean_24\\":0.0,\\"error5sum_rollingmean_24\\":0.0,\\"comp1sum\\":489.0,\\"comp2sum\\":549.0,\\"comp3sum\\":549.0,\\"comp4sum\\":564.0,\\"age\\":18.0}]}")')

# Compress the operationalization assets for easy blob storage transfer
MODEL_O16N = shutil.make_archive('o16n', 'zip', os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'])

# Create a new container if necessary, otherwise you can use an existing container.
# This command creates the container if it does not already exist. Else it does nothing.
az_blob_service.create_container(MODEL_CONTAINER, 
                                 fail_on_exist=False, 
                                 public_access=PublicAccess.Container)

# Transfer the compressed operationalization assets into the blob container.
az_blob_service.create_blob_from_path(MODEL_CONTAINER, "o16n.zip", str(MODEL_O16N) ) 


# Time the notebook execution. 
# This will only make sense if you "Run All" cells
toc = time.time()
print("Full run took %.2f minutes" % ((toc - tic)/60))

logger.log("Operationalization Run time", ((toc - tic)/60))

