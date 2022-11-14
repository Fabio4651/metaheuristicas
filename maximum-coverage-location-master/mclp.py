"""

Python implementation of the maximum coverage location problem.

The program randomly generates a set of candidate sites, among 
which the K optimal candidates are selected. The optimization 
problem is solved by integer programming. 

Author: Can Yang
Date: 2019-11-22

MIT License

Copyright (c) 2019 Can Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


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

def generate_candidate_sites_test(points,radius):
    '''
    teste onde será criado um circulo no mesmo lugar onde se encontra 1 ponto. com esta solução todos os candidatos têm 1 ponto disponível
    '''
    from shapely.geometry import Point
    sites = []
    for p in points:
        random_point = Point([p[0], p[1]])
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
    print(sites)

    # Nr de pontos / vertices
    I = points.shape[0]
    print('  Nr pontos (I) %g' % I)
    print(points)

    # Matrix with corresponding vertice/point to the distance of center of each candidate
    D = distance_matrix(points, sites)
    #print(D)
    condition = D<=radius
    #print(condition)
    D[condition]=1
    D[~condition]=0
    print("bool matrix")
    print(D)

    from gurobipy import Model, GRB, quicksum
    # Build model
    m = Model()
    # Add variables
    x = {}
    y = {}
    for i in range(I):
      y[i] = m.addVar(vtype=GRB.BINARY, name="y%d" % i)
    for j in range(J):
      x[j] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)

    m.update()

    #for j in range(J):
    #    print(j)

    # Add constraints | condicoes
    # condicao: necessários sempre K resultados (circulos / candidatos)
    m.addConstr(quicksum(x[j] for j in range(J)) == K)

    #print("debug 0")
    #print(y[0].getValue())
    #print("m.numVars", m.numVars)
    #print("m.objVal", m.objVal)
    #for v in m.getVars():
    #    print(v)
    #print("end debug 0")

    #print("debug 1")
    #for i in range(I):
    #    for j in np.where(D[i]==1)[0]:
    #        print(y[i])
    #        #print(j)
    #        #print("---")

    # TODO: tentar perceber esta restrição
    for i in range(I):
        m.addConstr(quicksum(x[j] for j in np.where(D[i]==1)[0]) >= y[i])

    # TODO: tentar perceber esta restrição
    m.setObjective(quicksum(y[i]for i in range(I)),GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    
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
    return opt_sites,m.objVal

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
Npoints = 30 #10
points,_ = make_moons(Npoints,noise=1)

#test = np.loadtxt('output.csv', delimiter = ',')
population = points
    
# Number of sites to select
K = 2

# Service radius of each site
radius = 0.35

# Candidate site size (random sites generated)
M = 10

#Npois = 30 e M = 1500 = working

# Run mclp 
# opt_sites is the location of optimal sites 
# f is the number of points covered
opt_sites,f = mclp(population,K,radius,M)

#print("sites: ")
#print(population)
#np.savetxt('output.csv', points, delimiter=',')
print("-end-")

# Plot the result 
plot_result(population,opt_sites,radius)   
 
"""
x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) """

plt.show()
