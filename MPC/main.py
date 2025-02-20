from utils import *
from MPC_Controller import *

def main():  
    parameters = read_config_file()

    # Add storage for predicted and original throttle values
    predicted_fuel_values = []
    original_fuel_values = []
    predicted_acc_values = []
    original_acc_values = []
    time_values = []

    # Load the vehicle data from Excel
    vehicle_data = Vehicle("./../Data/vehicle_data.csv")  # Replace with the actual file name
    controller = MPCController(parameters ,steps_ahead=10, dt=0.01)

    for _ in range(len(vehicle_data.data)):
        # Get the predicted throttle
        predicted_fuel,predicted_acc = controller.control(vehicle_data)

        # Append the values to the lists for plotting later
        predicted_fuel_values.append(predicted_fuel)
        original_fuel_values.append(vehicle_data.get_fuel())
        
        predicted_acc_values.append(predicted_acc)
        original_acc_values.append(vehicle_data.get_acceleration())
        
        time_values.append(vehicle_data.get_time())

        # Print the predicted throttle for debugging
        # print(f"Time: {vehicle_data.get_time():.2f}, Best Fuel Consumption: {predicted_fuel:.2f}")

        # Update to the next time step
        vehicle_data.update()

    # save_graph(time_values,original_fuel_values,predicted_fuel_values,"Fuel_comparison.png")
    # save_graph(time_values,original_acc_values,predicted_acc_values,"acc_comparison.png")
    save_graph_separated(time_values,original_fuel_values,predicted_fuel_values,"Fuel_comparison.png")
    save_graph_separated(time_values,original_acc_values,predicted_acc_values,"acc_comparison.png")
    
    save_predicted_throttle_to_csv(time_values,predicted_fuel_values,"predicted_fuel.csv")
    return


main()