import os
from pkg_resources import resource_filename

import numpy as np

import jiminy_py.core as jiminy  # The main module of jiminy - this is what gives access to the Robot
from jiminy_py.simulator import Simulator


data_root_path = resource_filename(
    "gym_jiminy.envs", "data/toys_models/simple_pendulum")
urdf_path = os.path.join(data_root_path, "simple_pendulum.urdf")

# Instantiate and initialize the robot
robot = jiminy.Robot()
robot.initialize(urdf_path, mesh_package_dirs=[data_root_path])

# Add a single motor
motor = jiminy.SimpleMotor("PendulumJoint")
robot.attach_motor(motor)
motor.initialize("PendulumJoint")

# Define the command: for now, the motor is off and doesn't modify the output torque.
def compute_command(t, q, v, sensors_data, command):
    command[:] = 0.0

# Instantiate and initialize the controller
controller = jiminy.ControllerFunctor()
controller.initialize(robot)

# Create a simulator using this robot and controller
simulator = Simulator(robot, controller)

# Set initial condition and simulation length
q0, v0 = np.array([0.1]), np.array([0.0])
simulation_duration = 10.0

# Launch the simulation
simulator.simulate(simulation_duration, q0, v0)