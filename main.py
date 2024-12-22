import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for space-time simulation
grid_size = 50  # Size of the grid
time_steps = 100  # Number of time steps
entropy_diffusion = 0.01  # Diffusion rate of entropy
order_growth = 0.005  # Growth rate of order
coupling_constant = 0.05  # Coupling between entropy and curvature

# Initialize fields
entropy = np.zeros((grid_size, grid_size))
order = np.zeros((grid_size, grid_size))
curvature = np.zeros((grid_size, grid_size))

# Set initial conditions
center = grid_size // 2
entropy[center, center] = 1.0  # Initial peak of entropy
order[center, center] = 0.5  # Initial peak of order

# Update fields over time
def update_fields():
    global entropy, order, curvature

    # Apply diffusion to entropy using a Laplacian operator
    entropy_next = np.copy(entropy)
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            laplacian_entropy = (
                entropy[i+1, j] + entropy[i-1, j] + entropy[i, j+1] + entropy[i, j-1] - 4 * entropy[i, j]
            )
            entropy_next[i, j] += entropy_diffusion * laplacian_entropy

    # Update order as entropy dissipates
    order_next = np.copy(order)
    order_next += order_growth * (1 - entropy / np.max(entropy))

    # Update curvature based on coupling between entropy and order
    curvature_next = coupling_constant * (entropy - order)

    # Update the fields globally
    return entropy_next, order_next, curvature_next

# Create an animation to show the dynamics
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def update_plot(frame):
    global entropy, order, curvature
    entropy, order, curvature = update_fields()

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    axs[0].imshow(entropy, extent=(-1, 1, -1, 1), cmap="plasma", origin="lower")
    axs[0].set_title("Entropy Field")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    axs[1].imshow(order, extent=(-1, 1, -1, 1), cmap="viridis", origin="lower")
    axs[1].set_title("Order Field")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")

    axs[2].imshow(curvature, extent=(-1, 1, -1, 1), cmap="coolwarm", origin="lower")
    axs[2].set_title("Curvature Field")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")

# Animation setup
ani = FuncAnimation(fig, update_plot, frames=time_steps, interval=100, repeat=True)

# Show the animation loop
plt.show()