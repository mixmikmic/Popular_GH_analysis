import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
# call our labelEncoder class
le = preprocessing.LabelEncoder()
# fit our data
le.fit([1, 2, 2, 6])
# print classes
le.classes_
# transform
le.transform([1, 1, 2, 6]) 
#print inverse data
le.inverse_transform([0, 0, 1, 2])

le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])

list(le.classes_)

le.transform(["tokyo", "tokyo", "paris"]) 

#list(le.inverse_transform([2, 2, 1]))



