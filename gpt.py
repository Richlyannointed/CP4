import math

# Define variables and constants
theta = math.pi / 4  # initial angular displacement (45 degrees)
omega = 0  # initial angular velocity
g = 9.81  # acceleration due to gravity (m/s^2)
L = 1  # length of pendulum (m)
m = 1  # mass of pendulum (kg)
dt = 0.01  # time step (s)
tmax = 10  # maximum simulation time (s)

# Initialize lists to store energy values over time
potential_energy = []
kinetic_energy = []
total_energy = []

# Perform simulation
t = 0
while t < tmax:
    # Calculate energy at current time step
    U = m * g * L * (1 - theta**2 / 2)
    K = (1/2) * m * L**2 * omega**2
    E = U + K
    potential_energy.append(U)
    kinetic_energy.append(K)
    total_energy.append(E)

    # Update theta and omega using Euler method
    theta_next = theta + omega * dt
    omega_next = omega - (g/L) * theta * dt
    theta = theta_next
    omega = omega_next

    t += dt

# Plot energy over time
import matplotlib.pyplot as plt

plt.plot(potential_energy, label='Potential energy')
plt.plot(kinetic_energy, label='Kinetic energy')
plt.plot(total_energy, label='Total energy')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Energy (J)')
plt.show()



