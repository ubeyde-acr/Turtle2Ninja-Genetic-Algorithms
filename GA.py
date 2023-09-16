import math
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from deap.tools import Statistics

from matplotlib.animation import FuncAnimation


def generate_maze(width, height, block_density):
    maze = np.ones((height, width), dtype=int)

    # Create walls around the maze
    maze[0, :] = 0
    maze[-1, :] = 0
    maze[:, 0] = 0
    maze[:, -1] = 0

    # Add random blocks
    num_blocks = int((width - 2) * (height - 2) * block_density)
    block_positions = np.random.randint(1, high=(width - 1), size=(num_blocks, 2))
    maze[block_positions[:, 0], block_positions[:, 1]] = 0

    exit_position = (
        width - 2,
        1,
    )  # Adjust the coordinates to be in the top-right corner
    maze[
        exit_position[1], exit_position[0]
    ] = 3  # Use a different value (e.g., 3) to represent the exit/prize

    # Place the starting position in the bottom-left corner
    start_position = (
        1,
        height - 2,
    )  # Adjust the coordinates to be in the bottom-left corner
    maze[
        start_position[1], start_position[0]
    ] = 2  # Use a unique value (e.g., 2) to represent the starting position

    return maze, start_position, exit_position


def plot_maze(maze):
    # Define colors for the exit/prize (e.g., red)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a color map with additional colors for the exit/prize and starting position
    custom_cmap = plt.cm.colors.ListedColormap(
        [WALLS_COLOR, GROUND_COLOR, START_COLOR, EXIT_COLOR]
    )

    # Display the maze with the custom color map
    im = ax.imshow(maze, cmap=custom_cmap, origin="upper")

    # Add a grid
    ax.grid(which="both", color="black", linewidth=1.2)

    # Set custom grid line intervals (1 pixel)
    ax.set_xticks(np.arange(-0.5, maze.shape[1] - 0.5, 1))
    ax.set_yticks(np.arange(-0.5, maze.shape[0] - 0.5, 1))

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Show the plot
    plt.savefig("maze.png", transparent=True)


def create_individual():
    individual = []

    # Determine the length of the genome for this individual (e.g., between 5 and 15)
    genome_length = random.randint(20, 100)

    # Generate the genome with random values (or directions in your case)
    for _ in range(genome_length):
        gene = random.choice(["left", "right", "up", "down"])
        individual.append(gene)
    return individual


# Define the evaluation function
def evaluate(individual):
    x, y = start_position
    steps = 0
    penalty1 = 0
    penalty2 = 0
    visited_positions = set()  # To keep track of visited positions

    for direction in individual:
        if direction == "left" and maze[y][x - 1] != 0:
            x -= 1
        elif direction == "right" and maze[y][x + 1] != 0:
            x += 1
        elif direction == "up" and maze[y - 1][x] != 0:
            y -= 1
        elif direction == "down" and maze[y + 1][x] != 0:
            y += 1
        else:
            penalty1 += 1
            continue

        steps += 1
        # Check if the new position has been visited before
        if (x, y) in visited_positions:
            penalty2 += 1  # You can adjust the penalty value as needed
        visited_positions.add((x, y))

    # Calculate the distance to the exit (Euclidean distance)
    dist = np.abs(exit_position[0] - x) + np.abs(exit_position[1] - y)
    wall_penalty = 200 * (penalty1 / len(individual))
    roll_back = 50 * (penalty2 / len(individual))
    return (dist + wall_penalty + roll_back,)


def update(individual):
    # Create a color map with additional colors for the exit/prize and starting position
    custom_cmap = plt.cm.colors.ListedColormap(
        [WALLS_COLOR, GROUND_COLOR, START_COLOR, EXIT_COLOR]
    )
    # Display the maze with the custom color map
    im = ax.imshow(maze, cmap=custom_cmap, origin="upper")
    # Add a grid
    ax.grid(which="both", color="black", linewidth=1.2)

    # Set custom grid line intervals (1 pixel)
    ax.set_xticks(np.arange(-0.5, maze.shape[1] - 0.5, 1))
    ax.set_yticks(np.arange(-0.5, maze.shape[0] - 0.5, 1))

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    x, y = start_position
    for ii, direction in enumerate(individual):
        if direction == "left":
            x -= 1
        elif direction == "right":
            x += 1
        elif direction == "up":
            y -= 1
        elif direction == "down":
            y += 1
        ax.add_patch(plt.Circle((x, y), 0.3, color="green", alpha=0.5))

        # Save the updated frame as an image for the current individual
        plt.savefig(
            os.path.join(individual_frames_dir, f"individual_{ii:04d}.png"),
            transparent=True,
        )


if __name__ == "__main__":
    RANDOM_SEED = 42
    # Number of generations and population size
    NUMBER_OF_GENERATIONS = 200
    POPULATION_SIZE = 1000
    MUTATION_RATE = 0.5
    MATING_RATE = 0.5
    PARENT_SELECT_COUNT = 200  # number of individuals to select for the next generation
    OFFSPRING_COUNT = 500  # The number of children to produce at each generation.
    TOURNAMENT_SIZE = 100

    EXIT_COLOR = "red"
    START_COLOR = "blue"
    WALLS_COLOR = "#D2691E"
    GROUND_COLOR = "#D3D3D3"

    # Set random seed for random module
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    maze, start_position, exit_position = generate_maze(15, 15, 0.2)
    plot_maze(maze)

    # Define the problem space
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    # Create a toolbox for creating individuals and evolving the population
    toolbox = base.Toolbox()

    toolbox.register(
        "individual", tools.initIterate, creator.Individual, create_individual
    )

    # Define the population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function
    toolbox.register("evaluate", evaluate)

    # Register the mate operation (crossover)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUTATION_RATE)

    # Register the select operation (parent selection)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Create the DEAP algorithm
    population = toolbox.population(n=POPULATION_SIZE)

    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run the genetic algorithm
    pop, logbook = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=PARENT_SELECT_COUNT,  # The number of individuals to select for the next generation
        lambda_=OFFSPRING_COUNT,  # The number of children to produce at each generation.
        cxpb=MATING_RATE,
        mutpb=MUTATION_RATE,
        ngen=NUMBER_OF_GENERATIONS,
        verbose=True,
        stats=stats,
        halloffame=hof,
    )
    # Retrieve the best individual
    best_individual = tools.selBest(pop, 1)[0]

    # Create a directory to save animation frames
    frames_dir = f"maze_animation_frames"
    if os.path.exists(frames_dir):
        # Use os.rmdir to remove an empty folder
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Iterate through the best individuals
    for i, ind in enumerate(hof):
        # Create a directory for each individual
        individual_frames_dir = os.path.join(frames_dir, f"individual_{i}")
        os.makedirs(individual_frames_dir, exist_ok=True)
        update(ind)
        ax.clear()

    # Close the animation figure
    plt.close()
