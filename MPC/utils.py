import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
import numpy as np

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

    def get_acceleration(self):
        return self.data.iloc[self.time_index]["Acceleration (m/s^2)"]

    def get_throttle(self):
        return self.data.iloc[self.time_index]["Throttle"]
    
    def get_fuel(self):
        return self.data.iloc[self.time_index]["Fuel CMEM (mg)"]

    def get_braking(self):
        return self.data.iloc[self.time_index]["Braking"]

    def get_steering(self):
        return self.data.iloc[self.time_index]["Steering"]

    def get_time(self):
        return self.data.iloc[self.time_index]["Step"]

def calculate_power(velocity, acceleration, parameters):
    F_mass = acceleration * parameters["mass"]
    F_rolling = parameters["rolling_coefficient"] * parameters["mass"] * parameters["g"]
    F_air = 0.5 * parameters["air_density"] * parameters["frontal_area"] * parameters["drag_coefficient"] *velocity**2
    F_total = F_mass + F_rolling + F_air
    return (F_total * velocity) / 1000 

def calculate_fuel_rate(parameters, P, N=35, V=3.6):
    """ CMEM fuel rate calculation """
    N0 = 30 * np.sqrt(3.0 / V)
    K = parameters["k"] * (1 + parameters["c"] * (N - N0))
    FR = (K * N * V + (P / parameters["eta"])) * (1 / parameters["lhv"])  # g/s
    return FR * 1000 * parameters["time_step"]  # mg per time step  

def read_config_file():
    # Load configuration from INI file
    config = configparser.ConfigParser()
    config.read("./MPC/config.ini")

    # Convert the section into a dictionary
    parameters = {key: float(value) for key, value in config["simulation_parameters"].items()}
    return parameters

def save_graph(time_values, original_throttle_values, predicted_throttle_values, filename="fuel_comparison.png"):
    # Create the outputs folder if it doesn't exist
    output_directory = "./MPC/outputs"
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


def save_predicted_throttle_to_csv(time_values, predicted_throttle_values, filename="predicted_fuel.csv"):
    # Create the outputs folder if it doesn't exist
    output_directory = "./MPC/outputs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    
    # Create a DataFrame with the time and throttle values
    data = {
        "Time (s)": time_values,
        "Predicted Fuel": predicted_throttle_values,
    }
    
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file in the outputs directory
    save_path = os.path.join(output_directory, filename)
    df.to_csv(save_path, index=False)

    print(f"Predicted Fuel values saved to: {save_path}")
    return