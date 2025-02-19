#!/usr/bin/env python

import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

# Define Constants
TIME_STEP = 0.05  # Simulation step (seconds)
HORIZON = 5  # Prediction horizon (adjust as needed)

# Vehicle parameters
mass = 1500  # kg
rolling_coefficient = 0.01
drag_coefficient = 0.3
air_density = 1.225  # kg/m^3
frontal_area = 2.9  # m^2
g = 9.81  # Gravity

# Fuel Model Constants
K0 = 0.2  # Base fuel consumption coefficient
C = 0.01  # Correction factor
eta = 0.3  # Engine efficiency
LHV = 42.5e6  # J/kg (lower heating value of fuel)

def calculate_power(speed, acceleration):
    """ Compute power demand """
    F_mass = acceleration * mass
    F_rolling = rolling_coefficient * mass * g
    # F_air = 0.5 * air_density * frontal_area * drag_coefficient * speed**2
    F_total = F_mass + F_rolling + 150
    return (F_total * speed) / 1000  # Convert to kW

def calculate_fuel_rate(P, N=35, V=3.6):
    """ CMEM fuel rate calculation """
    N0 = 30 * np.sqrt(3.0 / V)
    K = K0 * (1 + C * (N - N0))
    FR = (K * N * V + (P / eta)) * (1 / LHV)  # g/s
    return FR * 1000 * TIME_STEP  # mg per time step

def mpc_control(speed, acceleration):
    """ Solve MPC optimization problem for fuel-efficient acceleration """
    
    # Decision Variables
    a = cp.Variable(HORIZON)  # Acceleration (control input)
    v = cp.Variable(HORIZON + 1)  # Speed
    P = cp.Variable(HORIZON)  # Power consumption
    fuel = cp.Variable(HORIZON)  # Fuel consumption

    # Initial Conditions
    v0 = speed
    
    # Constraints and Objective
    constraints = [v[0] == v0]  # Start from current speed
    objective = 0

    for t in range(HORIZON):
        # Vehicle dynamics: v = v0 + a*t
        constraints += [v[t + 1] == v[t] + a[t] * TIME_STEP]
        
        # Compute power demand
        constraints += [P[t] >= calculate_power(v[t], a[t])]
        
        # Compute fuel rate
        constraints += [fuel[t] >= calculate_fuel_rate(P[t])]
        
        # Acceleration constraints (avoid high jerks)
        constraints += [a[t] >= -2, a[t] <= 2]
        
        # Speed constraints
        constraints += [v[t] >= 0, v[t] <= 30]  # Limiting to 30 m/s

        # Objective function: Minimize total fuel consumption
        objective += fuel[t]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.OSQP, verbose=True, max_iter=200000)

    if a.value is None:
        print("MPC solver failed to find a solution. Status:", problem.status)
        return 0  # Return zero acceleration if solver fails

    # Return optimal acceleration (first control input)
    return a.value[0]

# Load vehicle data
vehicle_data = pd.read_csv("vehicle_data.csv")

# Prepare lists for new acceleration and fuel consumption
optimized_accelerations = []
optimized_fuel_consumption = []

for index, row in vehicle_data.iterrows():
    speed = row['Speed (m/s)']
    acceleration = row['Acceleration (m/s^2)']
    
    # Get optimized acceleration
    new_acceleration = mpc_control(speed, acceleration)
    optimized_accelerations.append(new_acceleration)
    
    # Compute corresponding fuel consumption
    power = calculate_power(speed, new_acceleration)
    new_fuel_consumption = calculate_fuel_rate(power)
    optimized_fuel_consumption.append(new_fuel_consumption)

# Save results to a new CSV file
vehicle_data['Optimized Acceleration (m/s^2)'] = optimized_accelerations
vehicle_data['Optimized Fuel Consumption (mg)'] = optimized_fuel_consumption
vehicle_data.to_csv("optimized_vehicle_data.csv", index=False)