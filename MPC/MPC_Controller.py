from scipy.optimize import minimize
from utils import *

class MPCController:
    def __init__(self, parameters, steps_ahead=10, dt=0.01):
        self.steps_ahead = steps_ahead
        self.dt = dt
        self.bounds = [(parameters["max_deceleration"], parameters["max_acceleration"])] * steps_ahead  # Acceleration >= 0, Throttle 0 to 1
        self.parameters = parameters

    def objective(self, control_vars, init_state):
        cost = 0
        v, accel = init_state[0], init_state[1]  # Initial speed and acceleration

        total_distance = 0  # Total distance traveled (in meters)

        for t in range(self.steps_ahead):
            accel = control_vars[t]  # Predicted acceleration
            # throttle = control_vars[self.steps_ahead + t]  # Predicted throttle

            # Update speed using acceleration and calculate distance
            v += accel * self.dt
            # distance = v * self.dt  # Calculate distance traveled in this step
            # total_distance += distance  # Add to total distance

            power = calculate_power(v, accel, self.parameters)
            fuel_consumption = calculate_fuel_rate(self.parameters, power)

            # Acceleration change penalty
            acc_change_cost = abs(accel - (control_vars[t - 1] if t > 0 else 0.001))

            # Add costs for this distance
            cost += fuel_consumption + acc_change_cost

        # Normalize the total cost by the distance traveled to get cost per distance
        #cost_per_distance = cost / total_distance
        return cost


    def control(self, vehicle):
        # Initial state (speed, acceleration) from the Vehicle class
        init_state = (vehicle.get_speed(), vehicle.get_acceleration())
         
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