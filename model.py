# Load data
import numpy as np
import pandas as pd
data=pd.read_csv('C:/Users/shiny/Desktop/data_file/data.csv')

# Split data with stratified randomization
X=data.iloc[:,0:-1]
y=data.iloc[:,44]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,shuffle=True,random_state=42)
X_train_,X_validation,y_train_,y_validation=train_test_split(X_train,y_train,test_size=0.3,stratify=y_train,shuffle=True,random_state=42)

# Scale predictors with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_train_scaler_=scaler.transform(X_train_)
X_validation_scaler=scaler.transform(X_validation)
X_test_scaler=scaler.transform(X_test)

# Select optimal threshold
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_,y_train_)
y_pred_validation_proba=rf.predict_proba(X_validation)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def ROC(label, y_prob):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point

fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(y_validation, y_pred_validation_proba[:,1])

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# Refit the model in the training set
rf.fit(X_train_scaler,y_train)

# Make predictions in the test set
optimal_threshold=0.33
y_pred_proba=rf.predict_proba(X_test_scaler)
y_pred_list=[]
for i in range(len(y_test)):
    if (np.array(y_pred_proba)[:,1]>optimal_threshold)[i]==True:
        y_pred=1
    else:
        y_pred=0
    y_pred_list.append(y_pred)

# Evaluate model performance in test set
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

cm=confusion_matrix(y_test,y_pred_list)
baccuracy=balanced_accuracy_score(y_test,y_pred_list)
sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
f1=f1_score(y_test,y_pred_list)

print('balanced-accuracy:',baccuracy)
print('sensitivity:',sensitivity) 
print('specificity:',specificity)
print('F1-score:',f1)

# Evaluate feature importance with the built-in method
imp=rf.feature_importances_
print(imp)

# Evaluate feature importance with SHAP method
import shap
shap_values = shap.TreeExplainer(rf).shap_values(X_test_scaler)

%matplotlib inline
%config InlineBackend.figure_format='svg'
shap.plots.violin(shap_values[:,:,1], features=X_test_scaler, feature_names=X_test.columns, plot_type="layered_violin",max_display=20)
shap.summary_plot(shap_values[:,:,1],X_test_scaler,plot_type='bar',feature_names=X_test.columns,max_display=44,show=False)
