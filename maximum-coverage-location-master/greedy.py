# Import the necessary libraries
import math

# Define a function to check if a point is inside a circle
def is_point_in_circle(point, center, radius):
    x, y = point
    cx, cy = center
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius

# Define a function to find the circle that covers the most uncovered points
def find_best_circle(points, uncovered, centers, radius):
    best_circle = None
    max_coverage = 0
    for center in centers:
        coverage = 0
        for point in points:
            if point in uncovered and is_point_in_circle(point, center, radius):
                coverage += 1
        if coverage > max_coverage:
            best_circle = center
            max_coverage = coverage
    return best_circle

# Define a function to solve the maximum coverage problem
def solve_maximum_coverage(points, M, radius):
    # Initialize the list of uncovered points and the list of circle centers
    uncovered = points[:]
    centers = []

    # Repeatedly find the best circle and remove its coverage from the list of uncovered points
    for _ in range(M):
        center = find_best_circle(points, uncovered, centers, radius)
        if center is None:
            break
        centers.append(center)
        uncovered = [point for point in uncovered if not is_point_in_circle(point, center, radius)]

    # Return the resulting list of circle centers
    return centers
