import numpy as np
import math
import time
from numpy import random
from sklearn.datasets import make_moons as generate_pois
from matplotlib import pyplot
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

RADIUS = 0.3
POINTS_OF_INTEREST_SIZE = 200
POPULATION_SIZE = 20
NUM_ITERATIONS = 3000
NUM_CANDIDATES = 3

def generate_population(points):
    '''
    Generate initial population of Circles inside Hull
    Input:
        points: a Numpy array with shape of (N,2)
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    population = []
    while len(population) < POPULATION_SIZE:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            population.append(random_point)
    return np.array([(p.x,p.y) for p in population]), poly, hull

def fitness(point, points_of_interest):
    # Count the number of points of interest inside the circle
    count = 0
    for poi in points_of_interest:
        distance = math.sqrt((point[0] - poi[0]) ** 2 + (point[1] - poi[1]) ** 2)
        if distance <= RADIUS:
            count += 1
    # Return the number of points of interest inside the circle
    return count

def mutate(pol, point):
    min_x, min_y, max_x, max_y = pol.bounds
    while 1:
        mutation_value_x = random.uniform(min_x, max_x)
        mutation_value_y = random.uniform(min_y, max_y)
        random_point = Point([point[0] + mutation_value_x, point[1] + mutation_value_y])
        if (random_point.within(pol)):
            return random_point

def GA(pol, points_of_interest, population):
    # Measure the time taken to initialize the algorithm
    start_time = time.time()

    for i in range(NUM_ITERATIONS):
        # Evaluate the fitness of each point in the population
        fitness_values = [fitness(circle, points_of_interest) for circle in population]

        # Select the best point from the population
        best_point = population[fitness_values.index(max(fitness_values))]

        # Generate a new point by mutating the best point
        new_point = mutate(pol, best_point)

        # Replace the worst point in the population with the new point
        worst_point_index = fitness_values.index(min(fitness_values))
        population[worst_point_index] = new_point

    # Sort the population by fitness value
    sorted_population = sorted(population, key=lambda point: fitness(point, points_of_interest), reverse=True)

    # Measure the time taken to run the algorithm
    end_time = time.time()

    # Print the time taken to initialize the algorithm and run the algorithm
    #print("Initialization time:", start_time - time.time())
    print("Running time:", end_time - start_time)

    # Return the best N points from the final population
    best_points = sorted_population[:NUM_CANDIDATES]
    return best_points

def plot_result(points, population, candidates, hull):
    pyplot.figure(figsize=(8,8))
    pyplot.scatter(points[:,0],points[:,1],c='C0')
    ax = pyplot.gca()

    #hull draw
    #ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    #ax.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

    pyplot.scatter(population[:,0],population[:,1],c='C1',marker='+')
    for site in population:
        circle = pyplot.Circle(site, RADIUS, color='C1',fill=False,lw=2)
        ax.add_artist(circle)

    for site in candidates:
        circle = pyplot.Circle(site, RADIUS, color='green',fill=False,lw=2)
        ax.add_artist(circle)

    ax.axis('equal')
    ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=True, labeltop=False,
                       labelright=False, labelbottom=True)


# Generate random POI's
pois, y = generate_pois(POINTS_OF_INTEREST_SIZE, noise=1)

# Generate initial population(all circles)
population, polygon, hull = generate_population(pois)
candidates = GA(polygon, pois, population)
plot_result(pois, population, candidates, hull)

pyplot.show()