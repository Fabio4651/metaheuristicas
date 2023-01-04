import numpy as np
import math
from numpy import random
from sklearn.datasets import make_moons
from matplotlib import pyplot
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

RADIUS = 0.3
POPULATION_SIZE = 10
NUM_ITERATIONS = 5
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
    return np.array([(p.x,p.y) for p in population]), poly

def fitness(point, points_of_interest):
    # Count the number of points of interest inside the circle
    count = 0
    # Calculate the total distance from the point to all points of interest
    # total_distance = sum([math.sqrt((point[0] - poi[0]) ** 2 + (point[1] - poi[1]) ** 2) for poi in points_of_interest])
    # return total_distance
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

def pso(pol, points_of_interest, population):
    for i in range(NUM_ITERATIONS):
        # Evaluate the fitness of each point in the population
        fitness_values = [fitness(point, points_of_interest) for point in population]
        #print(fitness_values)

        # Select the best point from the population
        best_point = population[fitness_values.index(min(fitness_values))]

        # Generate a new point by mutating the best point
        new_point = mutate(pol, best_point)

        # Replace the worst point in the population with the new point
        worst_point_index = fitness_values.index(max(fitness_values))
        population[worst_point_index] = new_point

    # Return the best point from the final population
    # best_point = population[fitness_values.index(min(fitness_values))]
    # return best_point

    # Sort the population by fitness value
    sorted_population = sorted(population, key=lambda point: fitness(point, points_of_interest))

    # Return the best N points from the final population
    best_points = sorted_population[:NUM_CANDIDATES]
    return best_points

def plot_result(points, population, candidates):
    pyplot.figure(figsize=(8,8))
    pyplot.scatter(points[:,0],points[:,1],c='C0')
    ax = pyplot.gca()
    pyplot.scatter(population[:,0],population[:,1],c='C1',marker='+')
    for site in population:
        circle = pyplot.Circle(site, RADIUS, color='C1',fill=False,lw=2)
        ax.add_artist(circle)

    for site in candidates:
        circle = pyplot.Circle(site, RADIUS, color='#9467bd',fill=False,lw=2)
        ax.add_artist(circle)

    ax.axis('equal')
    ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=True, labeltop=False,
                       labelright=False, labelbottom=True)



# Generate random POI's
pois, y = make_moons(n_samples=100, noise=0.1)
#print(pois)

# Generate initial population(all circles)
population, polygon = generate_population(pois)

# Quantidade inicial de candidatos
#J = population.shape[0]
#print('Populacao inicial (J) %g' % J)

#print(population)
candidates = pso(polygon, pois, population)
#print(candidates)
#candidates = []
plot_result(pois, population, candidates)

# Plot the points on a scatter plot
#pyplot.scatter(X[:, 0], X[:, 1], c="#9467bd")
pyplot.show()