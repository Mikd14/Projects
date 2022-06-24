import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
import numpy as np
from sklearn.feature_selection import mutual_info_regression

data = pd.read_csv('train.csv')
pd.set_option("display.max_rows", None)

test_data= pd.read_csv('test.csv')
a = test_data['Id']
X_test = test_data.drop('Id', axis=1)



X = data.drop(['SalePrice', 'Id'], axis =1)
y = data.pop('SalePrice')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2,
                                                  random_state=1)

##label encodine for nominal values

dic = {
    'Street': {'Pave': 2, 'Grvl': 1},
    'Alley': {'Pave': 2, 'Grvl': 1, np.NaN:3}, # np.NaN: 0
    'LotShape': {'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0},
    #'LandContour': {'Lvl':3, 'Bnk':2, 'HLS':1, 'Low':0},
    'Utilities': {"AllPub": 4, "NoSewr": 3, "NoSeWa":2, "ELO": 1, np.NaN:0}, # np.NaN: 0
    'LandSlope': {"Gtl": 3, "Mod": 2, "Sev": 1},
    "ExterQual": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1},
    "ExterCond": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1}, 
    "BsmtQual": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.NaN:0}, # np.NaN: 0
    "BsmtCond": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.NaN:0}, # np.NaN: 0
    "BsmtExposure": {"Gd":4, "Av":3, "Mn":2, "No":1, np.NaN:0}, # np.NaN: 0
    "BsmtFinType1": {"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1, np.NaN:0}, # np.NaN: 0
    "BsmtFinType2": {"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1, np.NaN:0}, # np.NaN: 0
    #"HeatingQC": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1},
    "CentralAir": {"Y":1, "N":0},
    "Electrical": {"SBrkr":5, "FuseA":4, "FuseF":3, "FuseP":2, "Mix":1, np.NaN:0}, # np.NaN: 0
    "KitchenQual": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.NaN:0}, # np.NaN: 0
    "Functional": {"Typ":8, "Min1":7, "Min2":6, "Mod":5, "Maj1":4, "Maj2":3, "Sev":2, "Sal":1, np.NaN:0}, # np.NaN: 0
    "FireplaceQu": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.NaN:0}, # np.NaN: 0
    "GarageFinish": {"Fin":3, "RFn":2, "Unf":1, np.NaN:0}, # np.NaN: 0
    "GarageQual": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.NaN:0}, # np.NaN: 0
    #"GarageCond": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1,  np.NaN  :0}, # np.NaN: 0
    "PavedDrive": {"Y":2, "P":1, "N":0},
    "PoolQC":  {"Ex":4, "Gd":3, "TA":2, "Fa":1, np.NaN:0}, # np.NaN: 0
    }


X_train = X_train.replace(dic)
X_val = X_val.replace(dic)
X_test = X_test.replace(dic)


num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float64', 'int64']]
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']


for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c])

    
imputer = SimpleImputer(strategy='most_frequent')


X_train = pd.DataFrame(imputer.fit_transform(X_train))
X_val = pd.DataFrame(imputer.transform(X_val))
X_test = pd.DataFrame(imputer.transform(X_test))

X_train.columns = X.columns
X_val.columns = X.columns
X_test.columns = X.columns
### feature engineering
X_train['Living_SF'] = (X_train['GrLivArea'] + X_train['TotalBsmtSF'] +
                       X_train['1stFlrSF'] + X_train['2ndFlrSF'])


X_train['Living_SF'] = pd.to_numeric(X_train['Living_SF'])

X_val['Living_SF'] = (X_val['GrLivArea'] + X_val['TotalBsmtSF'] +
                       X_val['1stFlrSF'] + X_val['2ndFlrSF'])


X_val['Living_SF'] = pd.to_numeric(X_val['Living_SF'])

X_test['Living_SF'] = (X_test['GrLivArea'] + X_test['TotalBsmtSF'] +
                       X_test['1stFlrSF'] + X_test['2ndFlrSF'])


X_test['Living_SF'] = pd.to_numeric(X_test['Living_SF'])



#OHE 
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_train_cols = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]))
OH_val_cols = pd.DataFrame(encoder.transform(X_val[cat_cols]))
OH_test_cols = pd.DataFrame(encoder.transform(X_test[cat_cols]))

OH_train_cols.index = X_train.index
OH_val_cols.index = X_val.index
OH_test_cols.index = X_test.index

num_X_train = X_train.drop(cat_cols,axis=1)
num_X_val = X_val.drop(cat_cols,axis=1)
num_X_test = X_test.drop(cat_cols,axis=1)

OH_X_train = pd.concat([num_X_train, OH_train_cols], axis=1)
OH_X_val = pd.concat([num_X_val, OH_val_cols], axis=1)
OH_X_test = pd.concat([num_X_test, OH_test_cols], axis=1)

for c in OH_X_train.columns:
    OH_X_train[c] = pd.to_numeric(OH_X_train[c])

for c in OH_X_test.columns:
    OH_X_test[c] = pd.to_numeric(OH_X_test[c])

'''
###using gridCVsearch
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
max_depth = [2, 4, 6, 8]

param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
model = XGBRegressor()
grid =GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result =grid.fit(OH_X_train, y_train)

print(f'Best score is {grid_result.best_score_,}, using method {grid_result.best_params_}')
#results where: n_est:500, learningrate=0.1, maxdepth=2
'''
model = XGBRegressor(n_estimators=170, max_depth=2, objective='reg:squarederror',
                     random_state= 1, eval_metric='rmse')
model.fit(OH_X_train, y_train)
preds = model.predict(OH_X_test)
df = pd.DataFrame(preds, a)
df.to_csv('Submission.csv')

'''
results = (mean_squared_log_error(y_val, preds))**0.5


###trying to extract top 20 features into list
mi_scores = mutual_info_regression(OH_X_train, y_train)
mi_scores = pd.Series(mi_scores, name='MI Scores', index = OH_X_train.columns)
mi_scores = mi_scores.sort_values(ascending=False)
df = pd.DataFrame({'Feature':mi_scores.index, 'MI_score':mi_scores.values})

b = df.Feature.head(200)


lst = []
for i in b:

    lst.append(i)


new_x_train = OH_X_train[lst]
new_x_val = OH_X_val[lst]

model.fit(new_x_train, y_train)
preds = model.predict(new_x_val)
results = (mean_squared_log_error(y_val, preds))**0.5
'''



'''scores
no feat_eng = 0.147
with living_sf = 0.137
with outdoor spaces = 0.13671826799481382
top20 features = 0.1759366548991121
top 30 = 0.1544781001446269
top 40 = 0.1455618811495074
top 50 = 0.14666266639162345
top 60 = 0.14473141268722933
top 70 = 0.14468376914075937
top 100 = 0.14304954206918238
top 200 = 0.1415764557691001  - best
top 250 = 0.14405275962336667
'''
#print(get_mi_scores(X_train[num_cols], y_train))
