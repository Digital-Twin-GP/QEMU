from scipy.optimize import minimize, LinearConstraint
from utils import *

class MPCController:
    def __init__(self, mpc_parameters, fuel_parameters, time_step):
        self.steps_ahead = int(mpc_parameters["horizon"])
        self.dt = time_step
        self.bounds = [(mpc_parameters["max_deceleration"], mpc_parameters["max_acceleration"])] * self.steps_ahead  # Acceleration >= 0, Throttle 0 to 1
        self.MIN_SAFE_DISTANCE = mpc_parameters["min_safe_distance"]
        self.MAX_SAFE_DISTANCE = mpc_parameters["max_safe_distance"]
        self.Q_a = mpc_parameters["Q_a"]
        self.Q_v = mpc_parameters["Q_v"]
        self.Q_fuel = mpc_parameters["Q_fuel"]
        self.DELTA_MAX_ACC = mpc_parameters["delta_acc_max"]
        self.fuel_parameters = fuel_parameters

        # # Compute a dynamic acceleration limit (more restrictive when close to the vehicle)
        # adaptive_delta_acc_max = (distance_to_front_vehicle - min_safe_distance) / (max_safe_distance - min_safe_distance) 
        # adaptive_delta_acc_max = max(0.2, min(adaptive_delta_acc_max, delta_acc_max))  # Clamp between 0.2 and delta_acc_max
        # # Acceleration change constraint (updated dynamically)
        # delta_acc_matrix = np.eye(N) - np.eye(N, k=1)
        # constraint1 = LinearConstraint(delta_acc_matrix, -adaptive_delta_acc_max, adaptive_delta_acc_max)

        # Acceleration change constraint
        delta_acc_matrix = np.eye(self.steps_ahead) - np.eye(self.steps_ahead, k=1)
        constraint1 = LinearConstraint(delta_acc_matrix, -self.DELTA_MAX_ACC,self.DELTA_MAX_ACC)
        self.constraint1 = constraint1


    def objective(self, control_vars, init_state):
        cost = 0
        speed_t, acc_t, distance_t, MAX_ROAD_SPEED = init_state[0], init_state[1], init_state[2], init_state[3]

        for i in range(self.steps_ahead):
            fuel_cost = calculate_fuel(speed_t, acc_t, self.fuel_parameters, self.dt)

            if distance_t == -1:
                speed_ref = MAX_ROAD_SPEED
            else:
                # Ensure the distance stays within a reasonable range
                distance_clamped = max(self.MIN_SAFE_DISTANCE, min(distance_t, self.MAX_SAFE_DISTANCE))
                # Map distance to speed using a linear equation
                speed_ref = MAX_ROAD_SPEED * (distance_clamped - self.MIN_SAFE_DISTANCE) / (self.MAX_SAFE_DISTANCE - self.MIN_SAFE_DISTANCE)

            #check init_state[1] or control_vars[i-1] 
            cost += self.Q_v * (speed_ref - speed_t)**2  + self.Q_a * (control_vars[i] - init_state[1])**2 + self.Q_fuel * fuel_cost # + Q_a * acc_t**2

            acc_t = control_vars[i]
            if(distance_t != -1):
                distance_t -= speed_t * self.dt + 0.5 * acc_t * self.dt ** 2
            speed_t += acc_t * self.dt

        return cost



    def control(self, vehicle):
        # Initial state (speed, acceleration,distance) from the Vehicle class
        init_state = (vehicle.get_speed(), vehicle.get_acceleration(),vehicle.get_distance(), vehicle.get_max_speed())
         
        # Initial guess for control variables: moderate acceleration and throttle
        control_vars = (
            [vehicle.get_acceleration()] * self.steps_ahead   # Initial acceleration guess
        )

        # First acceleration step constraint
        first_element_matrix = np.zeros([self.steps_ahead, self.steps_ahead])
        first_element_matrix[0, 0] = 1
        constraint2 = LinearConstraint(first_element_matrix, init_state[1] - self.DELTA_MAX_ACC, init_state[1] + self.DELTA_MAX_ACC)
        
        # Run the optimization
        result = minimize(self.objective, control_vars, args=(init_state,), bounds=self.bounds,
                           constraints=[self.constraint1, constraint2], options={'maxiter': 1000})
        
        # Check optimization result
        optimized_acceleration = result.x[0]  # First optimized acceleration value
        optimized_velocity = vehicle.get_speed() + optimized_acceleration * self.dt
        fuel_consumption = calculate_fuel(optimized_velocity, optimized_acceleration, self.fuel_parameters, self.dt)

        return fuel_consumption , optimized_acceleration