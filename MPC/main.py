from utils import *
from MPC_Controller import *

def main():  
    fuel_parameters, mpc_parameters = read_config_files()

    # Add storage for predicted and original throttle values
    predicted_fuel_values = []
    original_fuel_values = []
    predicted_acc_values = []
    original_acc_values = []
    time_values = []

    # Load the vehicle data from Excel
    vehicle_data = Vehicle("./Data/vehicle_data.csv")  # Replace with the actual file name
    controller = MPCController(mpc_parameters, fuel_parameters, time_step=0.05)
    for i in range(len(vehicle_data.data)):
        # Get the predicted throttle
        predicted_fuel, predicted_acc = controller.control(vehicle_data)

        # Append the values to the lists for plotting later
        predicted_fuel_values.append(predicted_fuel)
        original_fuel_values.append(vehicle_data.get_fuel())
        predicted_acc_values.append(predicted_acc)
        original_acc_values.append(vehicle_data.get_acceleration())
        time_values.append(vehicle_data.get_time())
        
        # Update to the next time step
        vehicle_data.update()
        print(i)


    # save_graph(time_values,original_fuel_values,predicted_fuel_values,"Fuel_comparison.png")
    # save_graph(time_values,original_acc_values,predicted_acc_values,"acc_comparison.png")
    save_graph_separated(time_values,original_fuel_values,predicted_fuel_values,"Fuel_comparison.png")
    save_graph_separated(time_values,original_acc_values,predicted_acc_values,"acc_comparison.png")
    
    save_predictions_to_csv(time_values,original_acc_values,predicted_acc_values,
                                       original_fuel_values,predicted_fuel_values, vehicle_data.data,"predictions.csv")
    return


main()