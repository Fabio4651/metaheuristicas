import numpy as np
import math
import time
from numpy import random
from sklearn.datasets import make_moons as generate_pois
from matplotlib import pyplot
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point as PolygonPoint

RADIUS = 0.3
POINTS_OF_INTEREST_SIZE = 200
POPULATION_SIZE = 10
NUM_ITERATIONS = 100
NUM_CANDIDATES = 3

# Point class
class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

class Circle:
  def __init__(self, center : Point, radius):
    self.center = center
    self.radius = radius

  def overlaps(self, other):
    # Calculate the distance between the centers of the two circles
    distance = math.sqrt((self.center.x - other.center.x) ** 2 + (self.center.y - other.center.y) ** 2)

    # Check if the distance between the centers is less than the sum of the radius
    return distance < self.radius + other.radius

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
        random_point = PolygonPoint(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if (random_point.within(poly)):
            population.append(Circle(Point(random_point.x,random_point.y), RADIUS))
    return population, poly, hull

def fitness(circle, points_of_interest):
    # Count the number of points of interest inside the circle
    count = 0
    for poi in points_of_interest:
        distance = math.sqrt((circle.center.x - poi[0]) ** 2 + (circle.center.y - poi[1]) ** 2)
        if distance <= RADIUS:
            count += 1
    # Return the number of points of interest inside the circle
    return count

def mutate(pol, circle):
    min_x, min_y, max_x, max_y = pol.bounds
    mutation_value_x = random.uniform(min_x, max_x)
    mutation_value_y = random.uniform(min_y, max_y)
    random_point = PolygonPoint([circle.center.x + mutation_value_x, circle.center.y + mutation_value_y])

    new_circle = Circle(Point(random_point.x,random_point.y), RADIUS)

    # Check if the new circle is is the available search space
    if not (random_point.within(pol)):
        return mutate(pol, circle)

    # Check if the new circle overlaps with any other circles
    overlaps = any([c.overlaps(new_circle) for c in population if c != new_circle])
    if overlaps:
        return mutate(pol, circle)

    return new_circle

def GA(pol, points_of_interest, population):
    # Measure the time taken to initialize the algorithm
    start_time = time.time()

    for i in range(NUM_ITERATIONS):
        # Evaluate the fitness of each circle in the population
        fitness_values = [fitness(circle, points_of_interest) for circle in population]

        # Select the best circle from the population
        best_circle = population[fitness_values.index(max(fitness_values))]

        # Generate a new point by mutating the best circle
        new_circle = mutate(pol, best_circle)

        # Replace the worst circle in the population with the new circle
        worst_circle_index = fitness_values.index(min(fitness_values))
        population[worst_circle_index] = new_circle

    # Sort the population by fitness value
    sorted_population = sorted(population, key=lambda circle: fitness(circle, points_of_interest), reverse=True)

    # Measure the time taken to run the algorithm
    end_time = time.time()

    # Print the time taken to initialize the algorithm and run the algorithm
    #print("Initialization time:", start_time - time.time())
    print("Running time:", end_time - start_time)

    # Return the best N circles from the final population
    best_circles = sorted_population[:NUM_CANDIDATES]
    return best_circles

def plot_result(points, population, candidates, hull):
    pyplot.figure(figsize=(8,8))
    pyplot.scatter(points[:,0],points[:,1],c='C0')
    ax = pyplot.gca()

    #hull draw
    #ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    #ax.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

    #pyplot.scatter(population[:,0],population[:,1],c='C1',marker='+')
    for site in population:
        pyplot.scatter(site.center.x, site.center.y,c='C1',marker='+')
        circle = pyplot.Circle((site.center.x, site.center.y), RADIUS, color='C1',fill=False,lw=2)
        ax.add_artist(circle)

    for site in candidates:
        pyplot.scatter(site.center.x, site.center.y,c='green',marker='+')
        circle = pyplot.Circle((site.center.x, site.center.y), RADIUS, color='green',fill=False,lw=2)
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