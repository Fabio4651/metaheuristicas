import numpy as np
import math
#import time
from numpy import random
from sklearn.datasets import make_moons as generate_pois
from matplotlib import pyplot, patches as mpatches
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, Point as PolygonPoint

import matplotlib.animation as animation

RADIUS = 0.3
POINTS_OF_INTEREST_SIZE = 200
POPULATION_SIZE = 10
NUM_ITERATIONS = 100
NUM_CANDIDATES = 3

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

def generate_population():
    hull = ConvexHull(points_of_interest)
    polygon_points = hull.points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    population = []

    while len(population) < POPULATION_SIZE:
        random_point = PolygonPoint(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if (random_point.within(poly)):
            population.append(Circle(Point(random_point.x, random_point.y), RADIUS))
    return population, poly, hull

def fitness(circle):
    # Count the number of points of interest inside the circle
    count = 0
    for poi in points_of_interest:
        distance = math.sqrt((circle.center.x - poi[0]) ** 2 + (circle.center.y - poi[1]) ** 2)
        if distance <= RADIUS:
            count += 1
    # Return the number of points of interest inside the circle
    return count

def mutate(circle):
    min_x, min_y, max_x, max_y = polygon.bounds
    mutation_value_x = random.uniform(min_x, max_x)
    mutation_value_y = random.uniform(min_y, max_y)
    random_point = PolygonPoint([circle.center.x + mutation_value_x, circle.center.y + mutation_value_y])

    new_circle = Circle(Point(random_point.x,random_point.y), RADIUS)

    # Check if the new circle is in the available search space
    if not (random_point.within(polygon)):
        return mutate(circle)

    # Check if the new circle overlaps with any other circles
    overlaps = any([c.overlaps(new_circle) for c in population if c != new_circle])
    if overlaps:
        return mutate(circle)

    return new_circle

def GA():
    # Measure the time taken to initialize the algorithm
    #start_time = time.time()

    #for i in range(NUM_ITERATIONS):
    # Evaluate the fitness of each circle in the population
    fitness_values = [fitness(circle) for circle in population]

    # Select the best circle from the population
    best_circle = population[fitness_values.index(max(fitness_values))]

    # Generate a new circle by mutating the best circle
    new_circle = mutate(best_circle)

    # Replace the worst circle in the population with the new circle
    worst_circle_index = fitness_values.index(min(fitness_values))
    population[worst_circle_index] = new_circle

    # Sort the population by fitness value
    sorted_population = sorted(population, key=lambda circle: fitness(circle), reverse=True)

    # Measure the time taken to run the algorithm
    # end_time = time.time()

    # Print the time taken to initialize the algorithm and run the algorithm
    # print("Running time:", end_time - start_time)

    # Return the best N circles from the final population
    best_circles = sorted_population[:NUM_CANDIDATES]
    return sorted_population, best_circles

# Function to update the figure
def update(num):
    # Run the algorithm for one iteration
    temp_population, temp_candidates = GA()
    artists = []

    # Create new circle objects
    best_circles_circles = temp_candidates
    all_circles_circles = temp_population

    for site in all_circles_circles:
        marker = pyplot.scatter(site.center.x, site.center.y,c='C1',marker='+')
        circle = pyplot.Circle((site.center.x, site.center.y), RADIUS, color='C1',fill=False,lw=2)
        ax.add_artist(marker)
        ax.add_artist(circle)
        artists.append(marker)
        artists.append(circle)

    for site in best_circles_circles:
        marker = pyplot.scatter(site.center.x, site.center.y,c='green',marker='+')
        circle = pyplot.Circle((site.center.x, site.center.y), RADIUS, color='green',fill=False,lw=2)
        ax.add_artist(marker)
        ax.add_artist(circle)
        artists.append(marker)
        artists.append(circle)

    return artists

# Generate random POI's
#points_of_interest, y = generate_pois(POINTS_OF_INTEREST_SIZE, noise=1)
points_of_interest = np.loadtxt('./inputs/200_2.csv', delimiter = ',')

# Generate initial population(all circles)
population, polygon, hull = generate_population()
best_circles_circles = []
all_circles_circles = []

# Initialize the figure and axes
fig, ax = pyplot.subplots(figsize=(8,8))

# Plot the polygon in pyplot
#pyplot.fill(*polygon.exterior.xy)

# Define the x and y coordinates of the vertices of the square
#min_x, min_y, max_x, max_y = polygon.bounds
#x = [min_x, min_x, max_x, max_x]
#y = [min_y,max_y,max_y , min_y]

# Plot the square in pyplot
#pyplot.fill(x, y)

#pois_legend = mpatches.Patch(color='C0', label='Points of interest')
#population_legend = mpatches.Patch(color='C1', label='Population')
#candidates_legend = mpatches.Patch(color='green', label='Best candidates')

# Initialize the pois
#points_scatter = ax.scatter(points_of_interest[:,0],points_of_interest[:,1],c='C0')
pyplot.plot(points_of_interest[:,0], points_of_interest[:,1], 'o')
for simplex in hull.simplices:
    pyplot.plot(points_of_interest[simplex, 0], points_of_interest[simplex, 1], 'k-')

# Hull draw
ax.plot(points_of_interest[hull.vertices,0], points_of_interest[hull.vertices,1], 'r--', lw=2)
ax.plot(points_of_interest[hull.vertices[0],0], points_of_interest[hull.vertices[0],1], 'ro')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=NUM_ITERATIONS, blit=True)

# Add a legend
#ax.legend(handles=[pois_legend, population_legend, candidates_legend])
ax.axis('equal')
ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=True, labeltop=False,
                       labelright=False, labelbottom=True)

pyplot.show()