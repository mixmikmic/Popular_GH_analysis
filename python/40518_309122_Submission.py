from libtools import *

training = pd.read_csv('data-test.csv')

training.head()

training.describe()

training = training.fillna(-99999)

blind = pd.read_csv('blind.csv')

blind.head()

blind.describe()

training_SH = divisao_sh(training)
training_LM = divisao_lm(training)

blind_SH = divisao_sh(blind)
blind_LM = divisao_lm(blind)

training_SH.head()

training_LM.head()

blind_SH.head()

blind_LM.head()

X_SH = training_SH.drop(['Facies'],axis=1)
y_SH = training_SH['Facies']

X_LM = training_LM.drop(['Facies'],axis=1)
y_LM = training_LM['Facies']

X_SH_blind = blind_SH.drop(['Facies'],axis=1)
y_SH_blind = blind_SH['Facies']

X_LM_blind = blind_LM.drop(['Facies'],axis=1)
y_LM_blind = blind_LM['Facies']

from sklearn.model_selection import train_test_split

X_train_SH, X_test_SH, y_train_SH, y_test_SH = train_test_split(X_SH, y_SH, test_size=0.1)

X_train_LM, X_test_LM, y_train_LM, y_test_LM = train_test_split(X_LM, y_LM, test_size=0.1)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix

ETC_SH = ExtraTreesClassifier(n_estimators=500, bootstrap=True)
ETC_LM = ExtraTreesClassifier(n_estimators=500)

ETC_SH.fit(X_train_SH, y_train_SH)
ETC_LM.fit(X_train_LM, y_train_LM)

pred_SH = ETC_SH.predict(X_test_SH)
print(confusion_matrix(y_test_SH,pred_SH))
print(classification_report(y_test_SH,pred_SH))

pred_LM = ETC_LM.predict(X_test_LM)
print(confusion_matrix(y_test_LM,pred_LM))
print(classification_report(y_test_LM,pred_LM))

blind_pred_SH = ETC_SH.predict(X_SH_blind)
print(confusion_matrix(y_SH_blind, blind_pred_SH))
print(classification_report(y_SH_blind, blind_pred_SH))

blind_pred_LM = ETC_LM.predict(X_LM_blind)
print(confusion_matrix(y_LM_blind, blind_pred_LM))
print(classification_report(y_LM_blind, blind_pred_LM))

blind_pred_SH = pd.DataFrame(blind_pred_SH, index=X_SH_blind.index)
blind_pred_LM = pd.DataFrame(blind_pred_LM, index=X_LM_blind.index)
pred_blind = pd.concat([blind_pred_SH,blind_pred_LM])
pred_blind = pred_blind.sort_index()

y_blind = blind['Facies']

print(confusion_matrix(y_blind, pred_blind))
print(classification_report(y_blind, pred_blind))

training_data = pd.read_csv('training.csv')

training_data.head()

training_data.describe()

training_data_SH = divisao_sh(training_data)
training_data_LM = divisao_lm(training_data)

training_data_SH.describe()

training_data_LM.describe()

X_SH = training_data_SH.drop(['Facies'],axis=1)
y_SH = training_data_SH['Facies']

X_LM = training_data_LM.drop(['Facies'],axis=1)
y_LM = training_data_LM['Facies']

X_SH.describe()

X_LM.describe()

from sklearn.model_selection import train_test_split

X_train_SH, X_test_SH, y_train_SH, y_test_SH = train_test_split(X_SH, y_SH, test_size=0.1)

X_train_LM, X_test_LM, y_train_LM, y_test_LM = train_test_split(X_LM, y_LM, test_size=0.1)

ETC_SH = ExtraTreesClassifier(n_estimators=500, bootstrap=True)
ETC_LM = ExtraTreesClassifier(n_estimators=500)

ETC_SH.fit(X_train_SH, y_train_SH)
ETC_LM.fit(X_train_LM, y_train_LM)

pred_SH = ETC_SH.predict(X_test_SH)
print(confusion_matrix(y_test_SH,pred_SH))
print(classification_report(y_test_SH,pred_SH))

pred_LM = ETC_LM.predict(X_test_LM)
print(confusion_matrix(y_test_LM,pred_LM))
print(classification_report(y_test_LM,pred_LM))

validation = pd.read_csv('validation_data_nofacies.csv')

validation.head()

validation.describe()

validation['Label_Form_SH_LM'] = validation.Formation.apply((label_two_groups_formation))

validation.head()

validation_SH = divisao_sh(validation)
validation_LM = divisao_lm(validation)

validation_SH.head()

validation_LM.head()

X_val_SH = validation_SH.drop(['Formation','Well Name','Depth','NM_M'], axis=1)
X_val_LM = validation_LM.drop(['Formation','Well Name','Depth','NM_M'], axis=1)

X_val_SH.head()

X_val_LM.head()

pred_val_SH = ETC_SH.predict(X_val_SH)

pred_val_LM =ETC_LM.predict(X_val_LM)

pred_val_SH = pd.DataFrame(pred_val_SH, index=X_val_SH.index)
pred_val_LM = pd.DataFrame(pred_val_LM, index=X_val_LM.index)
pred_val = pd.concat([pred_val_SH,pred_val_LM])
pred_val = pred_val.sort_index()

pred_val.describe()

validation['Facies Pred'] = pred_val

validation=validation.drop(['Label_Form_SH_LM'],axis=1)

validation.head()

validation.to_csv('Prediction.csv')



