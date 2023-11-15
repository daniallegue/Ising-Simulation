from math import pi
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(size):
    """
    Initialize a random spin configuration for the Ising model.
    """
    return np.random.choice([-1, 1], size=(size, size))

def metropolis_step(lattice, beta=1.0):
    """
    Perform a Metropolis algorithm step for the Ising model.
    """
    #Generate sample
    i, j = randint(0, lattice.shape[0], size=2)  # Choose a random spin
    new_energy = 2 * lattice[i, j] * (lattice[(i + 1) % lattice.shape[0], j] +
                        lattice[i, (j + 1) % lattice.shape[1]] +
                        lattice[(i - 1) % lattice.shape[0], j] +
                        lattice[i, (j - 1) % lattice.shape[1]])

    #Flip spin in case of criteria condition met
    if new_energy < 0 or np.random.rand() < np.exp(-beta * new_energy):
        lattice[i, j] *= -1

def run_ising_model(size, steps, beta):
    """
    Run the Ising model simulation using the Metropolis algorithm.
    """
    lattice = initialize_lattice(size)
    history = [np.copy(lattice)]

    for _ in range(steps):
        metropolis_step(lattice, beta)
        history.append(np.copy(lattice))

    return np.array(history)

def plot_ising_model(history):
    """
    Plot the Ising model simulation.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(history.shape[0]):
        ax.imshow(history[i], cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"Step {i}")
        plt.pause(0.1)
        ax.clear()

    plt.show()

# Example usage:
size = 30
steps = 1000
beta = 1.2
simulation_history = run_ising_model(size, steps, beta)
plot_ising_model(simulation_history)

