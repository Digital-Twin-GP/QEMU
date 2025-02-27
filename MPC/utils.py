import pandas as pd
import matplotlib.pyplot as plt
import configparser
import numpy as np
import sys
import os
import math
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Fuel_Consumption_Calculation/Models')))
# from Random_Forest import randomForestPredict

class Vehicle:
    def __init__(self, excel_file):
        # Load the Excel file into a DataFrame
        self.data = pd.read_csv(excel_file)
        self.time_index = 0  # Start at the first row

    def update(self):
        # Move to the next row in the DataFrame (simulate a time step)
        if self.time_index < len(self.data) - 1:
            self.time_index += 1

    def get_speed(self):
        return self.data.iloc[self.time_index]["Speed (m/s)"]
    
    def get_max_speed(self):
        return self.data.iloc[self.time_index]["Allowed Speed (m/s)"]

    def get_acceleration(self):
        return self.data.iloc[self.time_index]["Acceleration (m/s^2)"]
    
    def get_distance(self):
        return self.data.iloc[self.time_index]["Distance (m)"]

    def get_time(self):
        return self.data.iloc[self.time_index]["Step"]
    
    def get_fuel(self):
        return self.data.iloc[self.time_index]["Fuel CMEM (mg)"]

def calculate_fuel(velocity, acceleration, fuel_parameters, time_step):
    # input = np.array([velocity,acceleration]).reshape(1, -1)
    # return randomForestPredict(input)[0]
    """ CMEM fuel rate calculation """
    F_mass = acceleration * fuel_parameters["mass"]
    F_rolling = fuel_parameters["rolling_coefficient"] * fuel_parameters["mass"] * fuel_parameters["g"]
    F_air = 0.5 * fuel_parameters["air_density"] * fuel_parameters["frontal_area"] * fuel_parameters["drag_coefficient"] * (velocity**2)
    F_total = F_mass + F_rolling + F_air
    P =(F_total * velocity) / 1000 
    N0 = 30 * math.sqrt(3.0 / fuel_parameters["V"])
    K = fuel_parameters["k"] * (1 + fuel_parameters["c"] * (fuel_parameters["N"] - N0))
    FR = (K * fuel_parameters["N"] * fuel_parameters["V"] + (P / fuel_parameters["eta"])) * (1 / fuel_parameters["lhv"]) * (1 + fuel_parameters["b"] * ((fuel_parameters["N"] - N0)**2))  # g/s
    return FR * 1000 * time_step  # mg per time step     


def read_config_files():
    config = configparser.ConfigParser()
    config.read("./MPC/mpc_config.ini")
    mpc_parameters = {
        "horizon": int(config["mpc_parameters"]["horizon"]),
        "delta_acc_max": float(config["mpc_parameters"]["delta_acc_max"]),
        "min_safe_distance": float(config["mpc_parameters"]["min_safe_distance"]),
        "max_safe_distance": float(config["mpc_parameters"]["max_safe_distance"]),
        "max_acceleration": float(config["mpc_parameters"]["max_acceleration"]),
        "max_deceleration": float(config["mpc_parameters"]["max_deceleration"]),
        "Q_v": float(config["mpc_parameters"]["Q_v"]),
        "Q_a": float(config["mpc_parameters"]["Q_a"]),
        "Q_fuel": float(config["mpc_parameters"]["Q_fuel"]),
    }

    config = configparser.ConfigParser()
    config.read("./MPC/fuel_config.ini")
    fuel_parameters = {
    "mass": float(config["fuel_parameters"]["mass"]),
    "rolling_coefficient": float(config["fuel_parameters"]["rolling_coefficient"]),
    "drag_coefficient": float(config["fuel_parameters"]["drag_coefficient"]),
    "air_density": float(config["fuel_parameters"]["air_density"]),
    "frontal_area": float(config["fuel_parameters"]["frontal_area"]),
    "g": float(config["fuel_parameters"]["g"]),
    "max_acceleration": float(config["fuel_parameters"]["max_acceleration"]),
    "max_deceleration": float(config["fuel_parameters"]["max_deceleration"]),
    "k": float(config["fuel_parameters"]["k"]),
    "c": float(config["fuel_parameters"]["c"]),
    "eta": float(config["fuel_parameters"]["eta"]),
    "lhv": float(config["fuel_parameters"]["lhv"]),
    "b": float(config["fuel_parameters"]["b"]),
    "N": float(config["fuel_parameters"]["N"]),
    "V": float(config["fuel_parameters"]["V"]),
    }


    return fuel_parameters, mpc_parameters

def save_graph(time_values, original_throttle_values, predicted_throttle_values, filename="fuel_comparison.png"):
    # Create the outputs folder if it doesn't exist
    output_directory = "./outputs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    
    # Plotting the curves
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, predicted_throttle_values, label="Predicted Fuel", color="blue")
    plt.plot(time_values, original_throttle_values, label="Original Fuel", color="orange", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Fuel")
    plt.title("Predicted Fuel vs Original Fuel")
    plt.legend()
    plt.grid(True)

    # Save the plot in the outputs directory
    save_path = os.path.join(output_directory, filename)
    plt.savefig(save_path, dpi=300)  # Save with high resolution (300 DPI)
    plt.close()  # Close the figure to free memory

    print(f"Graph saved as: {save_path}")
    return

def save_graph_separated(time_values, original_throttle_values, predicted_throttle_values, filename="fuel_comparison.png"):
    # Create the outputs folder if it doesn't exist
    output_directory = "./MPC/outputs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    num_points = 500
    total_data = len(time_values)

    if total_data > num_points:
        mid_idx = total_data // 2  # Find the middle index
        half_range = num_points // 2  # Half of the required points

        start_idx = max(0, mid_idx - half_range)
        end_idx = min(total_data, mid_idx + half_range)

        time_values = np.array(time_values)[start_idx:end_idx]
        original_throttle_values = np.array(original_throttle_values)[start_idx:end_idx]
        predicted_throttle_values = np.array(predicted_throttle_values)[start_idx:end_idx]

    # Create a figure with two subplots (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot predicted fuel
    axes[0].plot(time_values, predicted_throttle_values, label="Predicted Fuel", color="blue")
    axes[0].set_ylabel("Predicted Fuel")
    axes[0].set_title("Predicted Fuel over Time")
    axes[0].grid(True)
    axes[0].legend()

    # Plot original fuel
    axes[1].plot(time_values, original_throttle_values, label="Original Fuel", color="orange", linestyle="--")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Original Fuel")
    axes[1].set_title("Original Fuel over Time")
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    save_path = os.path.join(output_directory, filename)
    plt.savefig(save_path, dpi=300)  # Save with high resolution (300 DPI)
    plt.close()  # Close the figure to free memory

    print(f"Graph saved as: {save_path}")


def save_predictions_to_csv(time_values, original_acc_values, predicted_acc_values, original_fuel_values,predicted_fuel_values, vehicle, filename):
    # Create the outputs folder if it doesn't exist
    output_directory = "./MPC/outputs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    
    # Create a DataFrame with the time and throttle values
    data = {
        "Time (s)": time_values,
        "Distance": vehicle["Distance (m)"].tolist(),
        "Velocity": vehicle["Speed (m/s)"].tolist(),
        "Original Acc": original_acc_values,
        "Predicted Acc": predicted_acc_values,
        "Original Fuel": original_fuel_values,
        "Predicted Fuel": predicted_fuel_values,
    }
    
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file in the outputs directory
    save_path = os.path.join(output_directory, filename)
    df.to_csv(save_path, index=False)

    print(f"Prediction values saved to: {save_path}")
    return