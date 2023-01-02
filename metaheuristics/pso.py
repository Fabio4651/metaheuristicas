from random import randint
import time

from metaheuristics.circle import Circle

def fitness(circles, points):
  # Calculate the number of points inside each circle
  num_points_inside = [0] * len(circles)
  for p in points:
    for i, c in enumerate(circles):
      if p is inside c:
        num_points_inside[i] += 1
  
  # Return the sum of the number of points inside each circle
  return sum(num_points_inside)

def fitness_exclusive(circles, points):
  # Calculate the number of points inside each circle
  num_points_inside = [0] * len(circles)
  for p in points:
    inside_circle = False
    for i, c in enumerate(circles):
      if p is inside c:
        if not inside_circle:
          num_points_inside[i] += 1
          inside_circle = True
  
  # Return the sum of the number of points inside each circle
  return sum(num_points_inside)

WIDTH = 5
HEIGHT = 5
RADIUS = 0.5
POPULATION_SIZE = 200
NUM_GENERATIONS = 100
SELECTION_SIZE = 3

def point_swarm_optimization(points):
  # Measure the time taken to initialize the algorithm
  start_time = time.time()

  # Define the initial population of circles
  population = [Circle(randint(0, WIDTH), randint(0, HEIGHT), randint(0, RADIUS)) for _ in range(POPULATION_SIZE)]

  # Iterate over a fixed number of generations
  for i in range(NUM_GENERATIONS):
    # Evaluate the fitness of each individual in the population
    fitness_values = [c.fitness(points) for c in population]

    # Select the best individuals from the current population
    best_individuals = [population[i] for i in range(SELECTION_SIZE)]

    # Generate new individuals by mutating the best individuals
    new_individuals = [mutate(circles) for circles in best_individuals]

    # Replace the old population with the new population
    population = new_individuals

  # Measure the time taken to run the algorithm
  end_time = time.time()

  # Print the time taken to initialize the algorithm and run the algorithm
  print("Initialization time:", start_time - time.time())
  print("Running time:", end_time - start_time)

  # Return the best individual from the final population
  return max(population, key=lambda c: c.fitness(points))


def mutate(circles):
  # Select a random circle from the set of circles
  i = randint(0, len(circles) - 1)
  c = circles[i]
  
  # Generate a new circle that is slightly different from the selected circle
  new_x = c.x + randint(0, 1)
  new_y = c.y + randint(0, 1)
  new_r = c.r
  new_circle = Circle(new_x, new_y, new_r)
  
  # Replace the selected circle with the new circle
  circles[i] = new_circle
  
  return circles