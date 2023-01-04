
import numpy as np
from numpy import random
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

# radius of each individual
RADIUS = 0.5
POPULATION_SIZE = 200
POIS_SIZE = 20
NUM_GENERATIONS = 100

def distance_matrix(a,b):
    """
    Euclidean distance between two points
    Input:
        matrixA
        matrixB
    Return:
        matrix corresponding to a distance between the point on matrixA to the corresponding index on matrixB
    """
    from math import sqrt
    distance = lambda p1, p2: sqrt(((p1-p2)**2).sum())
    D = np.asarray([[distance(p1, p2) for p2 in b] for p1 in a])
    return D



def generate_population(points):
    '''
    Generate initial population of Circles inside Hull
    Input:
        points: a Numpy array with shape of (N,2)
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon, Point
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    population = []
    while len(population) < POPULATION_SIZE:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            population.append(random_point)
    return np.array([(p.x,p.y) for p in population])




def mclp(points):
    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    import time
    start = time.time()


    # Quantidade de candidatos
    J = sites.shape[0]
    print('  Candidatos (J) %g' % J)
    #print(sites)

    # Nr de pontos / vertices
    I = points.shape[0]
    print('N POIs (I) %g' % I)
    #print(points)

    # Matrix with corresponding vertice/point to the distance of center of each candidate
    D = distance_matrix(points, sites)
    #print(D)
    #neste ponto a matrix tem I entradas correspondentes a I pontos.
    #Cada ponto tem J valores, cada valor corresponde à distancia do ponto com o centro do circulo
    #se esta distancia for inferior à definida no radius, é alterada na matrix para valores bit 0 e 1
    condition = D<=radius
    #print(condition)
    D[condition]=1
    D[~condition]=0
    print("bool matrix")
    print(D)


    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objVal)
    
    solution = []
    if m.status == GRB.Status.OPTIMAL:
        #print("solution is optimal")
        for v in m.getVars():
            # print v.varName,v.x
            if v.x==1 and v.varName[0]=="x":
               solution.append(int(v.varName[1:]))
    opt_sites = sites[solution]
    return opt_sites

def plot_result(points, population):
    '''
    Plot the result
    Input:
        points: input points, Numpy array in shape of [N,2]
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        radius: the radius of circle
    '''
    plt.figure(figsize=(8,8))
    plt.scatter(points[:,0], points[:,1], c='C0')
    ax = plt.gca()
    plt.scatter(population[:,0], population[:,1], c='C1', marker='+')

    for site in population:
        circle = plt.Circle(site, RADIUS, color='C1', fill=False, lw=2)
        ax.add_artist(circle)

    ax.axis('equal')
    ax.tick_params(axis='both', left=False, top=False,
                    right=False, bottom=False, labelleft=True,
                    labeltop=False, labelright=False, labelbottom=True)
    


# read data from input
# or
# generate random points of interest(vertices / intersecções no nosso problema)
pois, y = make_moons(POIS_SIZE, noise=1)


population = generate_population(pois)

# Quantidade de candidatos
J = population.shape[0]
print('Populacao (J) %g' % J)

#population = mclp(pois)
#plot_result(pois, population)
plt.show()