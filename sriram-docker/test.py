
## import the needed libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd
import pickle
import xgboost as xgb


## function to display the test regression metrics
def regression_metrics(p,y,y_pred,train_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from math import sqrt
    scores = mean_squared_error(y,y_pred)
    scores = sqrt(scores)
    print(train_test,'RMSE: %.3f'% (scores))
    scores = mean_absolute_error(y,y_pred)
    print(train_test,'MAE: %.3f'% (scores))
    scores = r2_score(y,y_pred)
    print(train_test,'R2: %.3f'% (scores))
    adj_rsquared = 1 - (1-r2_score(y, y_pred)) * (len(y)-1)/(len(y)-p-1)
    print(train_test, "Adjusted-R2 : %.3f"% (adj_rsquared))
    print(train_test,'accuracy(<=3): %.3f'% (sum(abs(y-y_pred)<=3)/len(y)*100))


print("Predictions started..Please wait..")

## read the test dataset
df_test = pd.read_csv('test.csv')

## load the imputer model
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

## load the list of columns to be selected from the test dataset
with open('train_cols.pkl', 'rb') as f:
    train_cols = pickle.load(f)

## load the ML model
xg_model = xgb.XGBRegressor()
xg_model.load_model("xg_model_33.pkl")

## select only the required columns from the test dataset
df_test = df_test[train_cols]

## split into X & y datasets
df_test_copy = df_test.copy()
X = df_test_copy.drop('y',axis=1)
y = df_test_copy.pop('y')

## apply the Most frequent imputer
Xtrans = imputer.transform(X)

## re-establish the column names after the imputation
df_test_concat  = pd.concat([pd.DataFrame(Xtrans),pd.DataFrame(y)],axis=1)
df_test_concat.columns = df_test.columns
df_test = df_test_concat.copy()
df_test.head()

## split again in X&y and datasets
df_test_copy = df_test.copy()
X = df_test_copy.drop('y',axis=1)
y = df_test_copy.pop('y')

## predict using the ML model and display the regression metrics
best_model = xg_model
y_pred = best_model.predict(X)
regression_metrics(X.shape[1],y,y_pred,'Test') 

## write the predictions to a file
pd.DataFrame(y_pred).to_csv("predictions.csv",index=False,header=False)
print("Predictions written to predictions.csv")