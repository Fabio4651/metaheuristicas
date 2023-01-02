from random import randint


class Circle:
  def __init__(self, x, y, r):
    self.x = x
    self.y = y
    self.r = r
  
  def contains_point(self, point):
    return (point.x - self.x) ** 2 + (point.y - self.y) ** 2 <= self.r ** 2
  
  def mutate(self):
    self.x += randint(-1, 1)
    self.y += randint(-1, 1)
    self.r += randint(-1, 1)
  
  def fitness(self, points):
    return sum(1 for p in points if self.contains_point(p))
  