from scipy.optimize import minimize
from utils import *

class MPCController:
    def __init__(self, parameters, steps_ahead, dt):
        self.steps_ahead = steps_ahead
        self.dt = dt
        self.bounds = [(parameters["max_deceleration"], parameters["max_acceleration"])] * steps_ahead  # Acceleration >= 0, Throttle 0 to 1
        self.parameters = parameters

    def objective(self, control_vars, init_state):
        cost = 0
        v, accel, distance = init_state  
        
        prev_distance = distance

        # Open a log file in append mode
        with open("./MPC/mpc_log.txt", "a") as log_file:  
            for t in range(self.steps_ahead):
                accel = control_vars[t]  # Predicted acceleration

                # Update speed
                v = max(0, v + accel * self.dt)  # Prevent negative velocity

                # Distance traveled in this step
                dist = max(0, v * self.dt + 0.5 * accel * (self.dt) ** 2)

                # Adjusted distance cost to avoid division errors
                if distance == 11111:
                    distance_cost = 0
                else:
                    distance_cost = 5000 / (prev_distance - dist)

                prev_distance -= dist

                # Compute power and fuel consumption
                power = calculate_power(v, accel, self.parameters)
                fuel_consumption = calculate_fuel_rate(self.parameters, power)
                print("vel: ",v," accel: ",accel," power: ",power," fuel: ",fuel_consumption,"\n")

                # Penalize large acceleration changes
                acc_change_cost = 5000 * (accel - (control_vars[t - 1] if t > 0 else 0.001)) ** 2  

                # Penalize high acceleration to enforce efficiency
                #acc_penalty = 3000 * (accel ** 2)  

                # Total cost function
                cost += fuel_consumption + acc_change_cost + distance_cost  

                # Log to file
                log_entry = f"t: {t:.2f}, dist: {distance_cost:.2f}, fuel: {fuel_consumption:.2f}, acc: {acc_change_cost:.2f}\n"
                log_file.write(log_entry)

        return cost



    def control(self, vehicle):
        # Initial state (speed, acceleration) from the Vehicle class
        init_state = (vehicle.get_speed(), vehicle.get_acceleration(),vehicle.get_distance())
         
        # Initial guess for control variables: moderate acceleration and throttle
        control_vars = (
            # [vehicle.get_speed()] * self.steps_ahead +  # Initial velocity guess
            [vehicle.get_acceleration()] * self.steps_ahead   # Initial acceleration guess
            #[vehicle.get_throttle()] * self.steps_ahead  # Initial throttle guess
        )


        # Run the optimization
        result = minimize(self.objective, control_vars, args=(init_state,), bounds=self.bounds, method='SLSQP')

        # Check optimization result
        if result.success:
            #optimized_velocity = result.x[0]  # First optimized velocity value
            optimized_acceleration = result.x[0]  # First optimized acceleration value
            #optimized_throttle = result.x[2 * self.steps_ahead]  # First optimized throttle value
            vel = vehicle.get_speed() + optimized_acceleration * self.dt
            power = calculate_power(vel, optimized_acceleration, self.parameters)
            fuel_consumption = calculate_fuel_rate(self.parameters, power)
            return fuel_consumption , optimized_acceleration
        else:
            return -0.1,-0.1