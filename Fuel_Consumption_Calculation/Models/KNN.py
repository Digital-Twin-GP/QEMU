import sys
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocess import preprocessTrainSet

def knnTrain():
    # Preprocess the data to get training and testing data
    X_train, X_testValid, Y_train, Y_testValid = preprocessTrainSet()
    X = np.array(X_train)
    y = np.array(Y_train)

    # Create and train the KNN Regression model
    clf = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    clf.fit(X, y)

    # Save the trained model to a file
    pickle.dump(clf, open('./Fuel_Consumption_Calculation/Models/KNN_model.pkl', 'wb'))
    
    # Load the model and make predictions
    model = pickle.load(open('./Fuel_Consumption_Calculation/Models/KNN_model.pkl', 'rb'))

    # Predict on the training data
    y_pred_train = clf.predict(X_train)
    # Evaluate the model on training data
    mse_train = np.mean((y_pred_train - Y_train) ** 2)
    print(f"Train Data Mean Squared Error: {mse_train:.4f}\n")

    # Predict on the test data
    y_pred_test = model.predict(X_testValid)
    # Evaluate the model on test data
    mse_test = np.mean((y_pred_test - Y_testValid) ** 2)
    print(f"Test Data Mean Squared Error: {mse_test:.4f}\n")

    print("_________________________Training is completed_________________________")

def knnPredict(x_test):
    # Load the trained model
    model = pickle.load(open('./Fuel_Consumption_Calculation/Models/KNN_model.pkl', 'rb'))
    y_pred = model.predict(x_test)
    return y_pred