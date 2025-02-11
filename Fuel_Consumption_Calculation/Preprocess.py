import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocessTrainSet():
    # Load dataset
    file_path = "./Data/vehicle_data.csv"
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(columns=["Step"])

    # Check for missing values
    if df.isnull().sum().any():
        df = df.dropna()  # Remove rows with missing values

    # Define input (X) and output (y)
    X = df[["Fuel CMEM (mg)"]].values  # Input feature
    y = df["Fuel HBEFA (mg)"].values   # Target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # Print dataset shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


def preprocessTestSet():
    # Load dataset
    file_path = "./Data/test_data.csv"
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(columns=["Step"])

    # Check for missing values
    if df.isnull().sum().any():
        df = df.dropna()  # Remove rows with missing values

    # Define input (X) and output (y)
    X = df[["Fuel CMEM (mg)"]].values  # Input feature

    # Print dataset shapes
    print("X_test shape:", X.shape)

    return X