import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from  sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
import pickle as pkl

df= pd.read_csv(r"/home/user/Desktop/ml_internship/Crop Recommendation dataset.csv") 
print(df.head())

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()
knn = KNeighborsClassifier()

params_rf = {'n_estimators': [100, 200],'max_depth': [5, 10, 15]}
params_knn = {'n_neighbors': [3, 5, 7]}

grid_rf = GridSearchCV(rf,params_rf,cv=5)
grid_rf.fit(X_train, y_train)
print("Best RF Parameters:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_

grid_knn = GridSearchCV(knn,params_knn,cv=5)
grid_knn.fit(X_train, y_train)
print("Best KNN Parameters:", grid_knn.best_params_)
best_knn = grid_knn.best_estimator_

models = [best_rf, best_knn]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {model.__class__.__name__}: {score}")

pkl.dump(best_rf, open('random_forest.pkl', 'wb'))
pkl.dump(scaler, open('scaler.pkl', 'wb'))
pkl.dump(encoder, open('encoder.pkl', 'wb'))
print("Model, scaler, and encoder saved successfully.")
