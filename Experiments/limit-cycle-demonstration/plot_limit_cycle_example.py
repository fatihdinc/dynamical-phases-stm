import numpy as np
import matplotlib.pyplot as plt

# Define the grid in polar coordinates
rho_vals = np.linspace(0, 3, 20)  # Radial coordinate
theta_vals = np.linspace(0, 2 * np.pi, 50)  # Angular coordinate


# Convert to Cartesian coordinates
RHO, THETA = np.meshgrid(rho_vals, theta_vals)
X, Y = RHO * np.cos(THETA), RHO * np.sin(THETA)

# Compute the vector field components in polar coordinates
drho_dt = (1 - RHO**2) * RHO
dtheta_dt = 2 * np.pi *0.1 # Given r = .1

# Compute the Cartesian velocity components
dX = drho_dt * np.cos(THETA) - dtheta_dt * RHO * np.sin(THETA)
dY = drho_dt * np.sin(THETA) + dtheta_dt * RHO * np.cos(THETA)


# Normalize vectors for better visualization
magnitude = np.sqrt(dX**2 + dY**2)
dX /= magnitude
dY /= magnitude

# Plot the vector field
plt.figure(figsize=(7, 7))
plt.quiver(X, Y, dX, dY, alpha=0.6,scale = 20)

# Plot the limit cycle at ρ = 1
theta_cycle = np.linspace(0, 2 * np.pi, 100)
x_cycle = np.cos(theta_cycle)
y_cycle = np.sin(theta_cycle)
plt.plot(x_cycle, y_cycle, 'r', linewidth=2, label="Limit Cycle (ρ=1)")


plt.xlabel("X")
plt.ylabel("Y")
plt.title("Flow Map of the System in Cartesian Coordinates")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal')
plt.savefig('example_limit_cycle.pdf')
plt.show()