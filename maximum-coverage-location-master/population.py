import numpy as np
from numpy import random
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

def generate_candidates(points, M):
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
    #print(polygon_points)
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return hull, poly, np.array([(p.x,p.y) for p in sites])


def plot_result(points,opt_sites,radius, hull):
    #plt.figure(figsize=(8,8))
    fig, ax = plt.subplots(label="dist. populacao", nrows=3, ncols=1, figsize=(8, 20))
    ax[0].scatter(points[:,0],points[:,1],c='tab:blue')
    #ax = plt.gca()

    #hull draw
    ax[1].plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    ax[1].plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    ax[1].scatter(points[:,0],points[:,1],c='tab:blue')

    ax[2].scatter(points[:,0],points[:,1],c='tab:blue')
    ax[2].scatter(opt_sites[:,0],opt_sites[:,1],c='tab:purple',marker='x')
    for site in opt_sites:
        circle = plt.Circle(site, radius, color='tab:purple',fill=False,lw=2)
        ax[2].add_artist(circle)
    ax[2].plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    ax[2].plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    

    ax[0].axis('equal')
    ax[1].axis('equal')
    ax[2].axis('equal')
    
    ax[0].tick_params(axis='both',left=False, top=False, right=False,bottom=False, labelleft=True, labeltop=False,labelright=False, labelbottom=True)
    ax[1].tick_params(axis='both',left=False, top=False, right=False,bottom=False, labelleft=True, labeltop=False,labelright=False, labelbottom=True)
    ax[2].tick_params(axis='both',left=False, top=False, right=False,bottom=False, labelleft=True, labeltop=False,labelright=False, labelbottom=True)
    
# generate random distribution of points
numPoints = 30
population,_ = make_moons(numPoints,noise=0.15)
#points = np.loadtxt('output.csv', delimiter = ',')
#population = points

M = 200
hull, poly, candidates = generate_candidates(population, M)

r = 0.3

# Plot the result 
plot_result(population, candidates, r, hull)
 

plt.show()
