import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split  
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error, r2_score 
from  sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
import pickle as pkl

#load dataset 
df= pd.read_csv(r"/home/user/Desktop/ml_internship/Crop Recommendation dataset.csv") 
print(df.head()) 

import pandas as pd
import pickle as pkl

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(
    r"/home/user/Desktop/ml_internship/Crop Recommendation dataset.csv"
)

print(df.head())

# Encode target labels
encoder = LabelEncoder()

df['label'] = encoder.fit_transform(df['label'])

# Define predictors and target
X = df.drop('label', axis=1)

# Use Series instead of DataFrame
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
rf = RandomForestClassifier()
knn = KNeighborsClassifier()

# Hyperparameters
params_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15]
}

params_knn = {
    'n_neighbors': [3, 5, 7]
}

# GridSearch for Random Forest
grid_rf = GridSearchCV(
    rf,
    params_rf,
    cv=5
)

grid_rf.fit(X_train, y_train)

print("Best RF Parameters:", grid_rf.best_params_)

best_rf = grid_rf.best_estimator_

# GridSearch for KNN
grid_knn = GridSearchCV(
    knn,
    params_knn,
    cv=5
)

grid_knn.fit(X_train, y_train)

print("Best KNN Parameters:", grid_knn.best_params_)

best_knn = grid_knn.best_estimator_

# Evaluate Models
models = [best_rf, best_knn]

for model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    print(f"Accuracy of {model.__class__.__name__}: {score}")

# Save files
pkl.dump(best_rf, open('random_forest.pkl', 'wb'))

pkl.dump(scaler, open('scaler.pkl', 'wb'))

pkl.dump(encoder, open('encoder.pkl', 'wb'))

print("Model, scaler, and encoder saved successfully.")