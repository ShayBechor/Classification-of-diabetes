# Shay Bechor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import roc_curve,confusion_matrix,auc
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv('diabetes.csv')
data.head()
data.describe().T
data.info()
data.isnull().all()

data.hist(figsize=(20,14),bins=50)
plt.show()

print("Total zeros in SkinThickness(unreal situation) : ", data[data.SkinThickness == 0].shape[0])

plt.figure(figsize= (12,10))
sns.heatmap(data.corr(),annot=True)

plt.figure()
ax = sns.distplot(data['Glucose'][data.Outcome == 1], color ="darkturquoise", rug = True)
sns.distplot(data['Glucose'][data.Outcome == 0], color ="lightcoral", rug = True)
plt.legend(['Diabetes', 'No Diabetes'])
plt.title('Distrobution of Glucose through people with and without Diabetes')
plt.show()

# It looks like there are some outliers in BMI,BP,Glucose probably not entered value.
zerosBP = (data['BloodPressure']==0).sum()
zerosBMI = (data['BMI']==0).sum()
zerosGlu = (data['Glucose']==0).sum()
print("Zeros in BP = ", zerosBP ," Zeros in BMI = ",zerosBMI," Zeros in Glucose = ", zerosGlu )

print("1 - Diabetes, 0 - No Diabetes ")
print(data.Outcome.value_counts())

# now lets define our test and train data.
# we shall use sklearn to divide the data into test and train
# since our data is not really big, a good option is to use 80% as train and 20% as test
# stratify = y means that we are dividing the data with respect to how much of 1/0 in y.
x = data.drop(columns= 'Outcome')
y = data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1) #random = 1 so any time we get the same division

# Showing the dataset divison heads, we can see that X_train is without the Outcome column - the results
x_train.head()
y_train.head()


##KNN Using SKlearn Library to unscaled data

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Training Score:\n",knn.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))
print("F1 score is:\n",f1_score(y_test,y_pred))
print("Recall score is:\n",recall_score(y_test,y_pred))
print("Accuracy score is:\n",accuracy_score(y_test,y_pred))

confusion_matrix_knn = metrics.confusion_matrix(y_test,y_pred)
p = sns.heatmap(pd.DataFrame(confusion_matrix_knn), cmap="YlGnBu", annot=True,fmt='g')
plt.title('KNN Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


##hyper parametrs tuning to knn, unscaled data

# let us run over k=1 to 40 and see which one gives us the best accuracy.
accuracy_scores = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test,y_pred)*100)
print("the best n = ",np.argmax(accuracy_scores)+1," and the MAX accuracy is: ",max(accuracy_scores))

knn = KNeighborsClassifier(n_neighbors = np.argmax(accuracy_scores)+1)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
accuracy_score(y_test,y_pred)*100

confusion_matrix_knn = metrics.confusion_matrix(y_test,y_pred)
p = sns.heatmap(pd.DataFrame(confusion_matrix_knn), cmap="YlGnBu", annot=True,fmt='g')
plt.title('KNN Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# now we shall draw a ROC for KNN classifier:
y_prob_knn = knn.predict_proba(x_test)[:,1]
fprknn,tprknn,thr0 = roc_curve(y_test,y_prob_knn)
#calculate AUC
roc_auc_knn = auc(fprknn,tprknn)

plt.figure()
plt.plot(fprknn,tprknn,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for KNN')
plt.grid()
plt.show()

##Scaled Data Using Z tranformation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data_copy = data.copy(deep=True)
x_s =  pd.DataFrame(sc.fit_transform(data_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
y_s = data_copy.Outcome
x_s.head()
x_train_s,x_test_s,y_train_s,y_test_s = train_test_split(x_s,y_s,test_size=0.3,random_state=1) #random = 1 so any time we get the same division
x_test_s.head()

x_train_temp,x_test_temp,y_train_s,y_test_s = train_test_split(x,y,test_size=0.3,random_state=1) #random = 1 so any time we get the same division
mean_of_columns = x_train_temp.mean(axis=0)
std_of_columns = x_train_temp.std(axis=0)
x_test_s = (x_test_temp-mean_of_columns) / std_of_columns #scaling test with train mean and std
x_train_s = (x_train_temp-mean_of_columns) / std_of_columns
x_test_s.head()

x_train_s.hist(figsize=(20,14),bins=50)
plt.show()

knn = KNeighborsClassifier()
knn.fit(x_train_s,y_train_s)
y_pred = knn.predict(x_test_s)
knn.fit(x_train_s,y_train_s)
y_pred_s = knn.predict(x_test_s)
confusion_matrix_knn = metrics.confusion_matrix(y_test_s,y_pred_s)
p = sns.heatmap(pd.DataFrame(confusion_matrix_knn), cmap="YlGnBu", annot=True,fmt='g')
plt.title('KNN Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print("Classification Report is:\n",classification_report(y_test_s,y_pred_s))
print("Training Score:\n",knn.score(x_train_s,y_train_s)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test_s,y_pred_s))
print("R2 score is:\n",r2_score(y_test_s,y_pred_s))
print("F1 score is:\n",f1_score(y_test_s,y_pred_s))
print("Recall score is:\n",recall_score(y_test_s,y_pred_s))
print("Accuracy score is:\n",accuracy_score(y_test_s,y_pred_s))
print("roc score is:\n",roc_auc_score(y_test_s,y_pred_s))

# now we shall draw a ROC for KNN classifier:
y_prob_knns = knn.predict_proba(x_test_s)[:,1]
fprknn,tprknn,_ = roc_curve(y_test_s,y_prob_knns)

plt.figure()
plt.plot(fprknn,tprknn,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Data Scaled KNN')
plt.grid()
plt.show()

##found optimal K for accuracy


accuracy_scores_s = []
for i in range(1,70):
    knns = KNeighborsClassifier(n_neighbors=i)
    knns.fit(x_train_s,y_train_s)
    y_pred_knns = knns.predict(x_test_s)
    accuracy_scores_s.append(accuracy_score(y_test_s,y_pred_knns)*100)
print("the best n = ",np.argmax(accuracy_scores_s)+1," and the MAX accuracy is: ",max(accuracy_scores_s))
# after we checked a lot of options well save the model and the predictions.
knns = KNeighborsClassifier(n_neighbors= np.argmax(accuracy_scores_s)+1)
knns.fit(x_train_s,y_train_s)
y_pred_knns = knns.predict(x_test_s)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train_s,y_train_s)
y_pred_s = knn.predict(x_test_s)
confusion_matrix_knn = metrics.confusion_matrix(y_test_s,y_pred_s)
p = sns.heatmap(pd.DataFrame(confusion_matrix_knn), cmap="YlGnBu", annot=True,fmt='g')
plt.title('KNN Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Classification Report is:\n",classification_report(y_test_s,y_pred_s))
print("Training Score:\n",knn.score(x_train_s,y_train_s)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test_s,y_pred_s))
print("R2 score is:\n",r2_score(y_test_s,y_pred_s))
print("F1 score is:\n",f1_score(y_test_s,y_pred_s))
print("Recall score is:\n",recall_score(y_test_s,y_pred_s))
print("Accuracy score is:\n",accuracy_score(y_test_s,y_pred_s))

# now we shall draw a ROC for KNN classifier:
y_prob_knns = knn.predict_proba(x_test_s)[:,1]
fprknn,tprknn,_ = roc_curve(y_test_s,y_prob_knns)
#calculate AUC

plt.figure()
plt.plot(fprknn,tprknn,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Data Scaled KNN')
plt.grid()
plt.show()

##in another way to find the best hyper parameters - n neighbors
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,70)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid,cv=5,scoring='accuracy')
knn_cv.fit(x_train_s,y_train_s)

print("Best Score:" + str(knn_cv.best_score_*100))
print("Best Parameters: " + str(knn_cv.best_params_))

confusion_matrix_knn = metrics.confusion_matrix(y_test_s,y_pred_knns)
p = sns.heatmap(pd.DataFrame(confusion_matrix_knn), cmap="YlGnBu", annot=True,fmt='g')
plt.title('KNN Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# now we shall draw a ROC for KNN classifier:
y_prob_knns = knns.predict_proba(x_test_s)[:,1]
fprknn,tprknn,_ = roc_curve(y_test_s,y_prob_knns)
#calculate AUC
roc_auc_knn = auc(fprknn,tprknn)

plt.figure()
plt.plot(fprknn,tprknn,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Data Scaled KNN')
plt.grid()
plt.show()

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train_s, y_train_s)
#y_pred_rf = rf_opt.predict(x_test_s)
y_prob_knn = knn.predict_proba(x_test_s)[:,1]
fpr,tpr,thresh= roc_curve(y_test_s,y_prob_knn)
#cm = confusion_matrix(y_test_s, y_pred_rf)

gmean = []
for j in range(len(tpr)):
    gmean.append(GmeanCalc(fpr[j],tpr[j]))
bestthr = thresh[np.argmax(gmean)]
y_pred_knn = knn.predict_proba(x_test_s)[:,1]> bestthr

print(f'ROC AUC score: {roc_auc_score(y_test_s, y_pred_knn)}')
print('Accuracy Score: ',accuracy_score(y_test_s, y_pred_knn))
print('Fscore Score: ',f1_score(y_test_s, y_pred_knn))
print('recall Score: ',recall_score(y_test_s, y_pred_knn))
print('Highest Gmean : ',np.max(gmean))


# Drawing Confusion Matrix
confusion_matrix_knn = metrics.confusion_matrix(y_test_s,y_pred_knn)
p = sns.heatmap(pd.DataFrame(confusion_matrix_knn), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Optimal Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Optimal KNN')
plt.grid()
plt.show()

types=['rbf','linear','poly']
for i in types:
    svm_classifier = svm.SVC(kernel=i,probability=True)
    svm_classifier.fit(x_train_s,y_train_s)
    y_pred_svm = svm_classifier.predict(x_test_s)
  #  print("Classification Report is:\n",classification_report(y_test_s,y_pred_svm))
    print('accuracy for SVM kernel =',i,'is',metrics.accuracy_score(y_pred_svm,y_test_s)*100)

svm_classifier = svm.SVC(probability=True)
svm_classifier.fit(x_train_s,y_train_s)
y_pred_svm = svm_classifier.predict(x_test_s)
print('accuracy is',metrics.accuracy_score(y_pred_svm,y_test_s)*100)

confusion_matrix_svm = metrics.confusion_matrix(y_test_s,y_pred_svm)
# print("Classification Report is:\n",confusion_matrix_svm)
p = sns.heatmap(pd.DataFrame(confusion_matrix_svm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Default Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

y_prob_svm = svm_classifier.predict_proba(x_test_s)[:,1]
fpr,tpr,thresh_svm= roc_curve(y_test_s,y_prob_svm)
#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for SVM default model')
plt.grid()
plt.show()
print('AUC for SVM default','is',roc_auc)
print('Accuracy for SVM default','is',metrics.accuracy_score(y_pred_svm,y_test_s)*100)
print('Recall for SVM default','is',metrics.recall_score(y_pred_svm,y_test_s)*100)
print('F score for SVM default','is',metrics.f1_score(y_pred_svm,y_test_s)*100)

clf = svm.SVC()
parameters = { 'C':[0.1,0.2,1,2,5,10],'kernel':['linear','rbf','poly'],'degree':[1,2,3,4,5,6]}
grid = GridSearchCV(clf,parameters,cv=5)
grid_Result=grid.fit(x_train,y_train)
grid.best_params_

svm_clf_tuned = svm.SVC(kernel='linear',C=1,probability=True)
svm_clf_tuned.fit(x_train_s,y_train_s)
y_pred_svm_tuned = svm_clf_tuned.predict(x_test_s)
print('Accuracy for SVM tuned','is',metrics.accuracy_score(y_pred_svm_tuned,y_test_s)*100)
print('Recall for SVM tuned','is',metrics.recall_score(y_pred_svm_tuned,y_test_s)*100)
print('F score for SVM tuned','is',metrics.f1_score(y_pred_svm_tuned,y_test_s)*100)

confusion_matrix_svm = metrics.confusion_matrix(y_test_s,y_pred_svm_tuned)
# print("Classification Report is:\n",confusion_matrix_svm)
p = sns.heatmap(pd.DataFrame(confusion_matrix_svm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Tuned Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

y_prob_svm = svm_clf_tuned.predict_proba(x_test_s)[:,1]
fpr,tpr,thresh_svm= roc_curve(y_test_s,y_prob_svm)
#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for SVM Tuned model')
plt.grid()
plt.show()
print('AUC for SVM Tuned','is',roc_auc)
print('Accuracy for SVM Tuned','is',metrics.accuracy_score(y_pred_svm,y_test_s)*100)
print('Recall for SVM Tuned','is',metrics.recall_score(y_pred_svm,y_test_s)*100)
print('F score for SVM Tuned','is',metrics.f1_score(y_pred_svm,y_test_s)*100)

svm_clf_tuned.fit(x_train_s, y_train_s)
#y_pred_rf = rf_opt.predict(x_test_s)
y_prob_rf = svm_clf_tuned.predict_proba(x_test_s)[:,1]
fpr,tpr,thresh= roc_curve(y_test_s,y_prob_rf)
#cm = confusion_matrix(y_test_s, y_pred_rf)

gmean = []
for j in range(len(tpr)):
    gmean.append(GmeanCalc(fpr[j],tpr[j]))
bestthr = thresh[np.argmax(gmean)]
y_pred_rf = svm_clf_tuned.predict_proba(x_test_s)[:,1]> bestthr

print(f'ROC AUC score: {roc_auc_score(y_test_s, y_prob_rf)}')
print('Accuracy Score: ',accuracy_score(y_test_s, y_pred_rf))
print('Fscore Score: ',f1_score(y_test_s, y_pred_rf))
print('recall Score: ',recall_score(y_test_s, y_pred_rf))
print('Gmean Score: ',np.max(gmean))


# Drawing Confusion Matrix
confusion_matrix_rf = metrics.confusion_matrix(y_test_s,y_pred_rf)
p = sns.heatmap(pd.DataFrame(confusion_matrix_rf), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Optimal Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Optimal SVM')
plt.grid()
plt.show()

#Fitting RandomForestClassifier Model
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
y_pred_rf = rf_clf.predict(x_test)
y_prob_rf = rf_clf.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred_rf)

print(f'ROC AUC score: {roc_auc_score(y_test, y_prob_rf)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred_rf)*100)

# Drawing Confusion Matrix
confusion_matrix_rf = metrics.confusion_matrix(y_test,y_pred_rf)
p = sns.heatmap(pd.DataFrame(confusion_matrix_rf), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Random Forest Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Roc AUC Curve
fpr,tpr,_= roc_curve(y_test,y_prob_rf)
#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Random Forest')
plt.grid()
plt.show()

# now lets find the best parameters for Random Classifier and compare it to the "default" one.
FindBestParam_rf(x_train_s,y_train_s)

rf_opt = RandomForestClassifier(criterion='entropy',max_depth=86,max_features='auto',min_samples_leaf=8,
                                min_samples_split=10,n_estimators=440)


rf_opt = RandomForestClassifier(criterion='entropy',max_depth=35,max_features='auto',min_samples_leaf=6,
                                min_samples_split=5,n_estimators=294)

rf_opt.fit(x_train_s, y_train_s)
y_pred_rf = rf_opt.predict(x_test_s)
y_prob_rf = rf_opt.predict_proba(x_test_s)[:,1]
cm = confusion_matrix(y_test_s, y_pred_rf)

print(f'ROC AUC score: {roc_auc_score(y_test_s, y_prob_rf)}')
print('Accuracy Score: ',accuracy_score(y_test_s, y_pred_rf)*100)
print('Fscore Score: ',f1_score(y_test_s, y_pred_rf)*100)
print('recall Score: ',recall_score(y_test_s, y_pred_rf)*100)


# Drawing Confusion Matrix
confusion_matrix_rf = metrics.confusion_matrix(y_test_s,y_pred_rf)
p = sns.heatmap(pd.DataFrame(confusion_matrix_rf), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Random Forest Optimal Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Roc AUC Curve
fpr,tpr,thresh= roc_curve(y_test_s,y_prob_rf)
#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Optimal Random Forest')
plt.grid()
plt.show()

rf_opt.fit(x_train_s, y_train_s)
#y_pred_rf = rf_opt.predict(x_test_s)
y_prob_rf = rf_opt.predict_proba(x_test_s)[:,1]
fpr,tpr,thresh= roc_curve(y_test_s,y_prob_rf)
#cm = confusion_matrix(y_test_s, y_pred_rf)

gmean = []
for j in range(len(tpr)):
    gmean.append(GmeanCalc(fpr[j],tpr[j]))
bestthr = thresh[np.argmax(gmean)]
y_pred_rf = rf_opt.predict_proba(x_test_s)[:,1]> bestthr

print(f'ROC AUC score: {roc_auc_score(y_test_s, y_prob_rf)}')
print('Accuracy Score: ',accuracy_score(y_test_s, y_pred_rf))
print('Fscore Score: ',f1_score(y_test_s, y_pred_rf))
print('recall Score: ',recall_score(y_test_s, y_pred_rf))
print('Gmean Score: ',np.max(gmean))


# Drawing Confusion Matrix
confusion_matrix_rf = metrics.confusion_matrix(y_test_s,y_pred_rf)
p = sns.heatmap(pd.DataFrame(confusion_matrix_rf), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Random Forest Optimal Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC curve for Optimal Random Forest')
plt.grid()
plt.show()


importances = rf_opt.feature_importances_
forest_importances = pd.Series(importances,index = x.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances random forest")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

def FindBestParam_rf(x_train,y_train):
  rf = RandomForestClassifier()
  # Decision trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 40)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum levels in tree
  max_depth = [int(x) for x in np.linspace(10, 120, num = 40)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 4, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4,6,8]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}
  random_search = RandomizedSearchCV(rf,random_grid,cv=10)
  random_search.fit(x_train, y_train)
  print(random_search.best_estimator_)

def GmeanCalc(fpr,tpr):
    return np.sqrt(tpr*(1-fpr))