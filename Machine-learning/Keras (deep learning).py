
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#find % of missing values by column
#print(train.isnull().sum()/len(train)*100)

#find correlation between columns
print(train.corr())

for df in[train, test]:
    df.drop(('Soil_Type7', 'Soil_Type15','Soil_Type1'),inplace=True)

#feature engineering
train['total_distance_to_water'] = np.sqrt((train['Horizontal_Distance_To_Hydrology'])**2 + (train['Vertical_Distance_To_Hydrology'])**2)
train = train.drop(['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], axis=1)

test['total_distance_to_water'] = np.sqrt((test['Horizontal_Distance_To_Hydrology'])**2 + (test['Vertical_Distance_To_Hydrology'])**2)
test = test.drop(['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], axis=1)
#train['total_shade'] = train['Hillshade_9am'] + train['Hillshade_Noon'] + train['Hillshade_3pm']
train['mean_shade'] = (train['Hillshade_9am'] + train['Hillshade_Noon'] + train['Hillshade_3pm'])/3
train= train.drop(['Hillshade_9am','Hillshade_Noon', 'Hillshade_3pm'],axis=1)

test['mean_shade'] = (test['Hillshade_9am'] + test['Hillshade_Noon'] + test['Hillshade_3pm'])/3
test= test.drop(['Hillshade_9am','Hillshade_Noon', 'Hillshade_3pm'],axis=1)


#drop cat with 1 entry
train = train[train.Cover_Type != 5]

X = train.drop(['Cover_Type','Id'], axis=1)
le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(train.Cover_Type))
X_test = test.drop('Id', axis=1)



'''
over_strategy = { 4:500000, 3: 500000, 7:500000, 6: 500000}
oversample = SMOTE(sampling_strategy=over_strategy)
under_strategy = {2: 500000, 1:500000}
undersample = RandomUnderSampler(sampling_strategy=under_strategy)



steps = [('over', oversample), ('under', undersample)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(X, y)
'''


##Reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16',
                'float32', 'float64']
    for col in df.columns:
        if df[col].dtype=='bool':
            df[col] = df[col].astype(int)
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            #change int type to lowest poss
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

reduce_mem_usage(X)


score_list, test_pred_list, history_list = [], [], []

EPOCHS = 1
VERBOSE = 1
SINGLE_FOLD = False   
BATCH_SIZE = 2000
FOLDS = 10
RUNS = 1  # should be 1. increase the number of runs only if you want see how the result depends on the random seed

def my_model(X):

    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[-1])))
    model.add(Dense(26, activation='relu'))
    model.add(BatchNormalization())
   # model.add(Dropout(.1))
    #model.add(Dense(10, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(.1))
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

np.random.seed(1)
tf.random.set_seed(1)

for run in range(RUNS):
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y=y)):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        scaler = StandardScaler()

        X_tr = pd.DataFrame(scaler.fit_transform(X_tr))
        X_va = pd.DataFrame(scaler.transform(X_va))
        
        model = my_model(X_tr)

        #define callbacks
        lr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=5, verbose=VERBOSE)

        es = EarlyStopping(monitor='val_loss', patience=10, verbose=VERBOSE, mode='max', restore_best_weights=True)

        
        #train and save model
        history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_batch_size= len(X_va), callbacks=[lr,es], shuffle=True)

        history_list.append(history.history)
        model.save(f'model{run}.{fold}')

        #inference for validation after last epoch of fold
        y_va_pred = model.predict(X_va, batch_size=len(X_va))
        y_va_pred = np.argmax(y_va_pred, axis=1)

        #evaluation
        accuracy = accuracy_score(y_va, y_va_pred)

        print(f'Fold {run}.{fold} : Accuracy: {accuracy:.5f}')

        #test predicts
        test_pred_list.append(model.predict(Scaled_X_test), batch_size=BATCH_SIZE)

sub = pd.DataFrame()
sub['Id'] = test['Id']
sub['Cover_Type'] = np.argmax(sum(test_pred_list), axis=1)
sub.to_csv('submission.csv', index=False)




