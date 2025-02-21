from utils import *
from MPC_Controller import *

def main():  
    parameters = read_config_file()

    with open("./MPC/mpc_log.txt", "w") as log_file:
        log_file.write("")

    # Add storage for predicted and original throttle values
    predicted_fuel_values = []
    original_fuel_values = []
    predicted_acc_values = []
    original_acc_values = []
    time_values = []

    # Load the vehicle data from Excel
    vehicle_data = Vehicle("./Data/vehicle_data.csv")  # Replace with the actual file name
    controller = MPCController(parameters ,steps_ahead=1, dt=0.05)

    with open("./MPC/mpc_log.txt", "a") as log_file:  # Open file in append mode
        for _ in range(len(vehicle_data.data)):
            # Get the predicted throttle
            predicted_fuel, predicted_acc = controller.control(vehicle_data)
            # Append the values to the lists for plotting later
            predicted_fuel_values.append(predicted_fuel)
            original_fuel_values.append(vehicle_data.get_fuel())
            predicted_acc_values.append(predicted_acc)
            original_acc_values.append(vehicle_data.get_acceleration())
            time_values.append(vehicle_data.get_time())
            # Print the predicted throttle for debugging
            log_entry = f"Time: {vehicle_data.get_time():.2f}, Best Fuel Consumption: {predicted_fuel:.2f}, Predicted Acceleration: {predicted_acc:.2f}\n"
            # Write to the file
            log_file.write(log_entry)
            # Update to the next time step
            vehicle_data.update()


    # save_graph(time_values,original_fuel_values,predicted_fuel_values,"Fuel_comparison.png")
    # save_graph(time_values,original_acc_values,predicted_acc_values,"acc_comparison.png")
    save_graph_separated(time_values,original_fuel_values,predicted_fuel_values,"Fuel_comparison.png")
    save_graph_separated(time_values,original_acc_values,predicted_acc_values,"acc_comparison.png")
    
    save_predicted_throttle_to_csv(time_values,predicted_fuel_values,"predicted_fuel.csv")
    return


main()