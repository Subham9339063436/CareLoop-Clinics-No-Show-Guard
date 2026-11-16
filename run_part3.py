import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'appointments_clean.csv')
appointment_ids=df['appointment_id'].copy()
X=df[['age','distance_km','fee_inr','lead_days','prior_no_shows','reminder_sent','booking_channel','visit_type']].copy()
X['reminder_sent']=X['reminder_sent'].map({'Yes':1,'No':0})
X=pd.get_dummies(X,columns=['booking_channel','visit_type'])
y=df['no_show'].copy()
assert X.shape[1]==11, X.columns.tolist()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
train_ids=appointment_ids.loc[X_train.index]
test_ids=appointment_ids.loc[X_test.index]
baseline=float((y_test==0).mean())
print('X shape',X.shape,'train',len(X_train),'test',len(X_test),'test no-shows',int(y_test.sum()),'baseline',baseline)

# Task 3.2 Logistic regression
scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train_s,y_train)
lr_prob=lr.predict_proba(X_test_s)[:,1]
lr_pred=lr.predict(X_test_s)
lr_metrics={'accuracy':accuracy_score(y_test,lr_pred),'precision':precision_score(y_test,lr_pred,zero_division=0),'recall':recall_score(y_test,lr_pred,zero_division=0),'f1':f1_score(y_test,lr_pred,zero_division=0),'roc_auc':roc_auc_score(y_test,lr_prob),'confusion_matrix':confusion_matrix(y_test,lr_pred).tolist()}
coefs=dict(zip(X.columns,lr.coef_[0].tolist()))
rem_idx=list(X.columns).index('reminder_sent')
rem_coef=float(lr.coef_[0][rem_idx]); rem_sd=float(scaler.scale_[rem_idx])
rem_exp_scaled=float(np.exp(rem_coef)); rem_exp_real=float(np.exp(rem_coef/rem_sd))
print('LR metrics',lr_metrics)
print('LR coefs',coefs)
print('Reminder exp(coef)',rem_exp_scaled,'sd',rem_sd,'exp(coef/scale)',rem_exp_real)

# Task 3.3 trees
full_tree=DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train,y_train)
depth3=DecisionTreeClassifier(random_state=42,max_depth=3)
depth3.fit(X_train,y_train)
tree_metrics={
 'full_depth':{'train_accuracy':accuracy_score(y_train,full_tree.predict(X_train)),'test_accuracy':accuracy_score(y_test,full_tree.predict(X_test)),'depth':full_tree.get_depth()},
 'depth_3':{'train_accuracy':accuracy_score(y_train,depth3.predict(X_train)),'test_accuracy':accuracy_score(y_test,depth3.predict(X_test)),'depth':depth3.get_depth()}}
print('Tree metrics',tree_metrics)

# Task 3.4 random forest
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
rf_prob=rf.predict_proba(X_test)[:,1];rf_pred=rf.predict(X_test)
rf_metrics={'train_accuracy':accuracy_score(y_train,rf.predict(X_train)),'accuracy':accuracy_score(y_test,rf_pred),'recall':recall_score(y_test,rf_pred,zero_division=0),'roc_auc':roc_auc_score(y_test,rf_prob)}
importance=pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
print('RF metrics',rf_metrics,'top importance',importance.head(3).to_dict())
comparison=pd.DataFrame([
 {'model':'Logistic regression','accuracy':lr_metrics['accuracy'],'recall':lr_metrics['recall'],'roc_auc':lr_metrics['roc_auc']},
 {'model':'Decision tree (depth 3)','accuracy':accuracy_score(y_test,depth3.predict(X_test)),'recall':recall_score(y_test,depth3.predict(X_test),zero_division=0),'roc_auc':roc_auc_score(y_test,depth3.predict_proba(X_test)[:,1])},
 {'model':'Random forest (100 trees)','accuracy':rf_metrics['accuracy'],'recall':rf_metrics['recall'],'roc_auc':rf_metrics['roc_auc']},
])
print('Comparison\n',comparison.to_string(index=False))

# Task 3.5(a) Tune RF on training set only
param_grid={'n_estimators':[100,200],'max_depth':[3,5,8]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5,scoring='roc_auc')
grid.fit(X_train,y_train)
print('Grid best params',grid.best_params_,'best CV AUC',grid.best_score_)

# Task 3.5(b) Training-only CV AUC comparison. LR scaler is refit within each fold.
lr_pipe=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))])
cv_models={'Logistic regression':lr_pipe,'Decision tree depth 3':depth3,'Random forest':rf,'Tuned random forest':grid.best_estimator_}
cv_auc={name:float(cross_val_score(model,X_train,y_train,cv=5,scoring='roc_auc').mean()) for name,model in cv_models.items()}
print('Training CV AUC',cv_auc)
winner_name=max(cv_auc,key=cv_auc.get)
winner=cv_models[winner_name]
print('Winner',winner_name)

# Task 3.5(c) OOF threshold comparison
# OOF training probabilities for selected model only.
oof=cross_val_predict(winner,X_train,y_train,cv=5,method='predict_proba')[:,1]
threshold_rows=[]
for threshold in [0.5,0.35]:
 flagged=oof>=threshold
 threshold_rows.append({'threshold':threshold,'precision':precision_score(y_train,flagged,zero_division=0),'recall':recall_score(y_train,flagged,zero_division=0),'f1':f1_score(y_train,flagged,zero_division=0),'flagged':int(flagged.sum())})
threshold_table=pd.DataFrame(threshold_rows)
chosen_threshold=float(threshold_table.sort_values(['f1','threshold'],ascending=[False,False]).iloc[0]['threshold'])
print('OOF threshold metrics\n',threshold_table.to_string(index=False),'chosen',chosen_threshold)

# Task 3.5(d,e) Fit selected model on all training rows; single final test evaluation.
winner.fit(X_train,y_train)
test_prob=winner.predict_proba(X_test)[:,1]
test_flag=(test_prob>=chosen_threshold).astype(int)
test_final={'precision':precision_score(y_test,test_flag,zero_division=0),'recall':recall_score(y_test,test_flag,zero_division=0),'f1':f1_score(y_test,test_flag,zero_division=0),'flagged':int(test_flag.sum()),'roc_auc':roc_auc_score(y_test,test_prob),'threshold':chosen_threshold,'winner':winner_name}
print('Final test metrics',test_final)
risk=pd.DataFrame({'appointment_id':test_ids.values,'no_show':y_test.values,'risk_prob':test_prob,'flag':test_flag})
risk.to_csv(ROOT/'test_risk_scores.csv',index=False)

summary={
 'feature_columns':X.columns.tolist(),'feature_count':X.shape[1],'train_size':len(X_train),'test_size':len(X_test),'test_no_shows':int(y_test.sum()),'baseline_accuracy':baseline,
 'logistic':lr_metrics,'logistic_coefficients':coefs,'reminder_exp_scaled_coef':rem_exp_scaled,'reminder_standard_deviation':rem_sd,'reminder_exp_real_step':rem_exp_real,
 'trees':tree_metrics,'random_forest':rf_metrics,'rf_top_importance':importance.head(3).to_dict(),'comparison':comparison.to_dict(orient='records'),
 'grid_best_params':grid.best_params_,'grid_best_score':float(grid.best_score_),'cv_auc':cv_auc,'winner':winner_name,
 'oof_thresholds':threshold_table.to_dict(orient='records'),'chosen_threshold':chosen_threshold,'test_final':test_final,'risk_rows':len(risk)
}
(ROOT/'part3_summary.json').write_text(json.dumps(summary,indent=2))
