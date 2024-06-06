# Load the data used in this study
import numpy as np
import pandas as pd
data=pd.read_csv('C:/Users/Desktop/Disability_data.csv')

# Split data into train and test sets with stratified randomization
X=data.iloc[:,0:-1]
y=data.iloc[:,44]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,shuffle=True,random_state=42)

# Scale the predictors with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler().fit(X_trian)
X_train_scaler=min_max_scaler.transform(X_train)
X_test_scaler=min_max_scaler.transform(X_test)

# Define the random forest model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_scaler,y_train)

# Make predictions
# Make predictions with different thresholds
for t in range(0.01,1,0.01):
threshold=t
y_pred_proba=rf.predict_proba(X_test_scaler)
y_pred_list=[]
for i in range(len(y_test)):
    if (np.array(y_pred_proba)[:,1]>t)[i]==True:
        y_pred=1
    else:
        y_pred=0
    y_pred_list.append(y_pred)

# Evaluate model performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

cm=confusion_matrix(y_test,y_pred_list)
baccuracy=balanced_accuracy_score(y_test,y_pred_list)
sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
f1=f1_score(y_test,y_pred_list)

print('balanced-accuracy:',baccuracy)
print('recall:',sensitivity) 
print('specificity:',specificity)
print('F1-score:',f1)

# Evaluate feature importance with the built-in method
imp=rf.feature_importances_
print(imp)

# Evaluate feature importance with SHAP method
import shap
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_scaler,y_train)

shap_values = shap.TreeExplainer(rf).shap_values(X_train_scaler)

import matplotlib.pyplot as plt
fig=shap.summary_plot(shap_values[1],X_train_scaler,plot_type='bar',
                  feature_names=X_train.columns,max_display=44,show=False)
plt.savefig('C:/Users/shiny/Desktop/BADL失能预测/SHAP-Bar.jpg',dpi=900,bbox_inches='tight')


import matplotlib.pyplot as plt
shap.summary_plot(shap_values[1],X_train_scaler,plot_type='layered_violin',
                  feature_names=X_train.columns,max_display=44,show=False)
plt.savefig('C:/Users/Desktop/SHAP-layered-violin.jpg',dpi=900,bbox_inches='tight')

import matplotlib.pyplot as plt
shap.summary_plot(shap_values[1],X_train_scaler,plot_type='violin',
                  feature_names=X_train.columns,max_display=44,show=False)
plt.savefig('C:/Users/shiny/Desktop/BADL失能预测/SHAP-violin.jpg',dpi=900,bbox_inches='tight')


import matplotlib.pyplot as plt
shap.summary_plot(shap_values[1],X_train_scaler,plot_type='layered_violin',
                  feature_names=X_train.columns,max_display=20,show=False)
plt.title('(b) Top 20 predictors of disability by SHAP',fontsize=20)
plt.savefig('C:/Users/shiny/Desktop/BADL失能预测/SHAP-layered-violin-20.jpg',dpi=900,bbox_inches='tight')

import matplotlib.pyplot as plt
shap.summary_plot(shap_values[1],X_train_scaler,plot_type='violin',
                  feature_names=X_train.columns,max_display=20,show=False)
plt.savefig('C:/Users/shiny/Desktop/BADL失能预测/SHAP-violin-20.jpg',dpi=900,bbox_inches='tight')



shap.summary_plot(shap_values[1],X_train_scaler,plot_type='dot',
                  feature_names=X_train.columns,max_display=44)



























explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_scaler)
shap.summary_plot(shap_values, X_train_scaler, plot_type="bar",
                  feature_names=X_train.columns,max_display=20)


shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

#local interpretation
shap.force_plot(explainer.expected_value, shap_values[84,:], X.iloc[84,:],matplotlib=True)
print(y[84])

shap.summary_plot(shap_values[:,0],X,alpha=1)

shap.summary_plot(shap_values, X, plot_type="bar")

















import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_scaler)




from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_scaler,y_train)
shap_values = shap.TreeExplainer(rf).shap_values(X_train_scaler)
shap.summary_plot(shap_values, X_train_scaler,feature_names=X.columns,max_display=10,plot_type='beeswarm')


shap.plots.waterfall(shap_values[0], max_display=14)



shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)



















from sklearn.preprocessing import MinMaxScaler
X_scaler=MinMaxScaler().fit_transform(X)


#Random forest
imp_data=np.empty((44,))
for i in range(100):
    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(X_scaler,y)
    
    imp=rf.feature_importances_
    imp_data=np.vstack((imp_data,imp))







#LASSO
from sklearn.linear_model import LassoCV
lasso=LassoCV(eps=0.005,n_alphas=100,cv=10)
lasso.fit(X,y)

imp=lasso.coef_

---------------------------------------------
from sklearn.linear_model import LassoCV
rf=LassoCV(eps=0.005,n_alphas=100,cv=10)
rf.fit(X_train_scaler,y_train)

#evaluate model performance (repeated bootstrap)
optimal_threshold=0.35
y_pred_proba=rf.predict_proba(X_test_scaler)
y_pred_list=[]
for i in range(len(y_test)):
    if (np.array(y_pred_proba)[:,1]>optimal_threshold)[i]==True:
        y_pred=1
    else:
        y_pred=0
    y_pred_list.append(y_pred)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import f1_score
   
cm=confusion_matrix(y_test,y_pred_list)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
baccuracy=balanced_accuracy_score(y_test,y_pred_list)
sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
ppv=cm[1,1]/(cm[1,1]+cm[0,1])
G_mean=np.sqrt(sensitivity*specificity)
fpr,tpr,thresholds=roc_curve(y_test,y_pred_proba[:,1])
auroc=auc(fpr,tpr)
f1=f1_score(y_test,y_pred_list)
_precision,_recall,threshold=precision_recall_curve(y_test,y_pred_proba[:,1])
auprc=auc(_recall,_precision)

print('accuracy:',accuracy)  
print('balanced-accuracy:',baccuracy)
print('recall:',sensitivity) 
print('specificity:',specificity)
print('ppv:',ppv)   #np.nanmean(ppv_list)
print('Mean g-mean:',G_mean) 
print('F1-score:',f1)
print('Mean auroc:',auroc)
print('AUPRC:',auprc)
