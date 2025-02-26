from Models.Random_Forest import *
from Models.SVR import *
from Models.Linear_Regression import *
from Models.KNN import *
from Preprocess import preprocessTestSet
import pandas as pd

def train():
    randomForestTrain()
    svrTrain()
    linearRegressionTrain()
    knnTrain()


def predict():
    testSet = preprocessTestSet()
    randomForestPrediction = []
    SVRPrediction = []
    linearRegressionPrediction = []
    KNNPrediction = []
    for x in testSet:
        # Make the prediction using the reshaped input
        input = np.array([x]).reshape(1, -1)
        randomForestPrediction.append(randomForestPredict(input)[0])
        SVRPrediction.append(svrPredict(input)[0])
        linearRegressionPrediction.append(linearRegressionPredict(input)[0])
        KNNPrediction.append(knnPredict(input)[0])

    return randomForestPrediction,SVRPrediction,linearRegressionPrediction,KNNPrediction
    


def main(type):
    if type == 1: #Train Models
        train()
    elif type == 2: #Predict models to compare predictions
        file_path = "./Fuel_Consumption_Calculation/test_data.csv"
        df = pd.read_csv(file_path)
        randomForestPrediction,SVRPrediction,linearRegressionPrediction,KNNPrediction = predict()
        df["Random Forest"] = randomForestPrediction
        df["SVR"] = SVRPrediction
        df["Linear Regression"] = linearRegressionPrediction
        df["KNN"] = KNNPrediction
        df.to_csv(file_path, index=False)

main(2)