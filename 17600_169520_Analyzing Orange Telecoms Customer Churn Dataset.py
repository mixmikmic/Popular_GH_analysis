get_ipython().magic('matplotlib inline')

import pandas as pd
import seaborn as sns
import plotly.plotly as py
import cufflinks as cf

# Import dataset with spark CSV package
orange_sprk_df = sqlContext.read.load("../_datasets_downloads/churn-bigml-80.csv",
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')

orange_final_dataset = sqlContext.read.load("../_datasets_downloads/churn-bigml-20.csv",
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')

# Print Dataframe Schema. That's DataFrame = Dataset[Row]
orange_sprk_df.cache()
orange_sprk_df.printSchema()

# Display first 5 Rows or Spark Dataset
orange_sprk_df.toPandas().head()

num_set = orange_sprk_df.describe().toPandas().transpose()
num_set.head()

# Display the numeric index
num_set.index.values

# Drop the `summary` and `Area code` columns and slice the dataframe using the numeric index.
new_df = orange_sprk_df.toPandas()
new_df = new_df[num_set.index.drop(['summary','Area code'])]
new_df.head()

axs = pd.scatter_matrix(new_df, figsize=(18,18))

# Rotate axis labels and remove axis ticks
n = len(new_df.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
    

binary_map = {'Yes':1.0, 'No':0.0, 'True':1.0, 'False':0.0}


# Remove correlated and unneccessary columns
col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 'Total night charge','Total intl charge']
orange_df = orange_sprk_df.toPandas().drop(col_to_drop, axis=1)

# Change categorical data to Numeric for the traininfg set 80%
orange_df[['International plan', 'Voice mail plan']] = orange_df[['International plan', 'Voice mail plan']].replace(binary_map)
orange_df['Churn'] = orange_df['Churn'].apply(lambda d: d.astype(float))


# Perform same function for the 20% test data
orange_train_df = orange_final_dataset.toPandas().drop(col_to_drop, axis=1)
orange_train_df[['International plan', 'Voice mail plan']] = orange_train_df[['International plan', 'Voice mail plan']].replace(binary_map)
orange_train_df['Churn'] = orange_train_df['Churn'].apply(lambda d: d.astype(float))

# Print sample
orange_df.head()

orange_2sparkdf = sqlContext.createDataFrame(orange_df)
orange_2sparkdf.take(2)


from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics

# Separate the features and the target variable

def labelData(data):
    """
        LabeledPoint(self, label, features)
        Label: row[end] i.e. 'Churn'
        Features: row[0 : end-1] i.e. other columns beside last column
    """
    return data.map(lambda row: LabeledPoint(row[-1], row[:-1]))

# Example: Target Variable and Feature Variables.
labelData(orange_2sparkdf).takeSample(True, 5)

# Divide into training data and test data
training_data, testing_data = labelData(orange_2sparkdf).randomSplit([0.8, 0.2])

# Design the model
# Map categorical variables to number of into categories.
# index 1 `International plan` has 2 variables (Yes/No) and Index 2 'Voice mail plan' has 2 variables (Yes/No)
model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                     categoricalFeaturesInfo={1:2, 2:2}, 
                                     maxDepth=3, impurity='gini', maxBins=32
                                    )
print(model.toDebugString())

print(orange_df.columns[4])
print(orange_df.columns[12])
print(orange_df.columns[6])
print(orange_df.columns[1])

def getPredictionLabels(model, testing_data):
    predictions = model.predict(testing_data.map(lambda r: r.features))
    return predictions.zip(testing_data.map(lambda r: r.label))

def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print('Precision of True ', metrics.precision(1))
    print('Precision of False', metrics.precision(0))
    print('Recall of True / False Positive  ', metrics.recall(1))
    print('Recall of False / False Negative  ', metrics.recall(0))
    print('F-1 Score        \n\n ', metrics.fMeasure())
    print(pd.DataFrame([['True Positive','False Negative'],['False Positive','True Negative']]))
    print('\nConfusion Matrix \n\n {}'.format(metrics.confusionMatrix().toArray()))
    

predictions_and_labels = getPredictionLabels(model, testing_data)
predictions_and_labels.take(5)

printMetrics(predictions_and_labels)

orange_2sparkdf.groupBy('Churn').count().toPandas()

# Sample all the 1s (100% of ones) amd 20% of zeros.
strat_orange_2sparkdf = orange_2sparkdf.sampleBy('Churn', fractions={0:0.2, 1:1.0})
strat_orange_2sparkdf.groupBy('Churn').count().toPandas()

training_data, testing_data = labelData(strat_orange_2sparkdf).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                     categoricalFeaturesInfo={1:2, 2:2}, 
                                     maxDepth=3, impurity='gini', maxBins=32 )

predictions_and_labels = getPredictionLabels(model, testing_data)
printMetrics(predictions_and_labels)



