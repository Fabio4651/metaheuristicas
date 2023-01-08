import csv
import random
from sklearn.datasets import make_moons as generate_pois

randfile = open("./inputs/200_2.csv", "w", newline='')
writer = csv.writer(randfile, delimiter=",")

POINTS_OF_INTEREST_SIZE = 200
points_of_interest, y = generate_pois(POINTS_OF_INTEREST_SIZE, noise=1)

for point in points_of_interest:
    row = point[0],point[1]
    writer.writerow(row)
    #print(row)