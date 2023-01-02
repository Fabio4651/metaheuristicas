
import numpy as np
from numpy import random
#from scipy.spatial import distance_matrix
#from gurobipy import Model, GRB, quicksum
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

def generate_candidate_sites(points,M=100):
    '''
    Generate M candidate sites with the convex hull of a point set
    Input:
        points: a Numpy array with shape of (N,2)
        M: the number of candidate sites to generate
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon, Point
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

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

def mclp(points,K,radius,M):
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

    # Gerar os M possíveis candidatos iniciais.
    # TODO: alterar nome da variable "sites" para candidates
    sites = generate_candidate_sites(points, M)
    #sites = generate_candidate_sites_test(points, radius)

    # Quantidade de candidatos
    J = sites.shape[0]
    print('  Candidatos (J) %g' % J)
    #print(sites)

    # Nr de pontos / vertices
    I = points.shape[0]
    print('  Nr pontos (I) %g' % I)
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
    #as linhas do matrix nao podem ser alteradas pois correspondem aos valores de cada ponto
    #as colunas de cada linha corresponde à distancia de cada ponto com 1 circulo

    


    
    solution = []
    solution.append(1)
    solution.append(2)
    opt_sites = sites[solution]
    return opt_sites

def plot_result(points,opt_sites,radius):
    '''
    Plot the result
    Input:
        points: input points, Numpy array in shape of [N,2]
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        radius: the radius of circle
    '''
    plt.figure(figsize=(8,8))
    plt.scatter(points[:,0],points[:,1],c='C0')
    ax = plt.gca()
    plt.scatter(opt_sites[:,0],opt_sites[:,1],c='C1',marker='+')
    for site in opt_sites:
        circle = plt.Circle(site, radius, color='C1',fill=False,lw=2)
        ax.add_artist(circle)
    ax.axis('equal')
    ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=True, labeltop=False,
                       labelright=False, labelbottom=True)
    
# generate random distribution of points
Npoints = 10 #10
points,_ = make_moons(Npoints,noise=1)

population = points
    
# Number of sites to select
K = 2

# Service radius of each site
radius = 0.35

# Candidate site size (random sites generated)
M = 5 #nr de circulos gerados inicialmente

opt_sites = mclp(population,K,radius,M)


# Plot the result 
plot_result(population,opt_sites,radius)   

plt.show()
