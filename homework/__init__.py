import pandas as pd 
test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip")
train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip")
test_data['Age']=2021-test_data['Year']
train_data['Age']=2021-train_data['Year']
test_data=test_data.drop(columns=['Year','Car_Name'])
train_data=train_data.drop(columns=['Year','Car_Name'])

x_train=train_data.drop(columns="Present_Price")
y_train=train_data["Present_Price"]
x_test=test_data.drop(columns="Present_Price")
y_test=test_data["Present_Price"]

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.feature_selection import f_regression,SelectKBest

categorical_features=['Fuel_Type','Selling_type','Transmission']
numerical_features= [col for col in x_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('scaler',MinMaxScaler(),numerical_features),
    ],
)

pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ('feature_selection',SelectKBest(f_regression)),
        ('classifier', LinearRegression())
    ]
)


from sklearn.model_selection import GridSearchCV


param_grid = {
    'feature_selection__k':range(1,15),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]

}
model=GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    )

model.fit(x_train, y_train)

import pickle
import os
import gzip

models_dir = '../files/models'
os.makedirs(models_dir, exist_ok=True)
compressed_model_path = "../files/models/model.pkl.gz"

with gzip.open(compressed_model_path, "wb") as file:
    pickle.dump(model, file)

import json
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,median_absolute_error

def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': float(r2_score(y_train, y_train_pred)),
        'mse': float(mean_squared_error(y_train, y_train_pred)),
        'mad': float(median_absolute_error(y_train, y_train_pred))
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': float(r2_score(y_test, y_test_pred)),
        'mse': float(mean_squared_error(y_test, y_test_pred)),
        'mad': float(median_absolute_error(y_test, y_test_pred)),
    }
    output_dir = '../files/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:  
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

calculate_and_save_metrics(model, x_train, x_test, y_train, y_test)



