import numpy as np
import scipy.spatial as sc


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def subtract(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

# calculates the cross product of vector p1 and p2
# if p1 is clockwise from p2 wrt origin then it returns +ve value
# if p2 is anti-clockwise from p2 wrt origin then it returns -ve value
# if p1 p2 and origin are collinear then it returs 0


def cross_product(p1, p2):
    return p1.x * p2.y - p2.x * p1.y


def direction(p1, p2, p3):
    return cross_product(p3.subtract(p1), p2.subtract(p1))


def left(p1, p2, p3):
    return direction(p1, p2, p3) < 0


# checks if p3 makes right turn at p2
def right(p1, p2, p3):
    return direction(p1, p2, p3) > 0


# checks if p1, p2 and p3 are collinear
def collinear(p1, p2, p3):
    return direction(p1, p2, p3) == 0


def on_segment(p1, p2, p):
    return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)


# checks if line segment p1p2 and p3p4 intersect
def intersect(p1, p2, p3, p4):
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False

def fun_list_calc_ordered_nodes_by_distance(nodes_list, x_array = None, y_array = None):
    for i in nodes_list:
    #     print(f'node {i}, id {i.node_id}, x {i.x_cord}, ordered by dist {i.ordered_nodes_by_distance}')
        i.calc_ordered_nodes_by_distance(x_array,y_array)
    # print('fun list nodes printing')
    # for i in nodes_list:
    #     print(f'node {i}, id {i.node_id}, x {i.x_cord}, ordered by dist {i.ordered_nodes_by_distance}')
    return nodes_list

def fun_list_calc_ordered_nodes_by_distance_2(node_class):
    array_x_y = np.array((node_class.x_coords, node_class.y_coords)).T
    distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
    node_class.ordered_nodes_by_distance_matrix = np.argsort(distances)
    for i, node in enumerate(node_class.nodes_list):
        node.distance_to_nodes = distances[i]
        node.ordered_nodes_by_distance = node_class.ordered_nodes_by_distance_matrix[i]


class Nodes:
    nodes_list = []
    y_coords = np.array([])
    x_coords = np.array([])
    total_nodes = 0

    def __init__(self, x_cord, y_cord):
        Nodes.nodes_list.append(self)
        Nodes.x_coords = np.append(Nodes.x_coords, x_cord)
        Nodes.y_coords = np.append(Nodes.y_coords, y_cord)

        self.node_id = Nodes.total_nodes

        self.x_cord = x_cord
        self.y_cord = y_cord

        self.ordered_nodes_by_distance =[]

        Nodes.total_nodes += 1

    def distance(self, other_node):
        return np.linalg.norm([self.x_cord-other_node.x_cord ,self.y_cord-other_node.y_cord])

    def calc_ordered_nodes_by_distance(self, x_array = None, y_array = None):
        if x_array is None:
            x_array = self.x_coords
            y_array = self.y_coords
        # print('calc ord x:', self.x_cord-Nodes.x_coords)
        # print(print('calc ord y:', self.x_cord-Nodes.y_coords))
        # print(f'class coords: {self.x_coords}')
        # print(f'object coords: {self.x_cord}')
        distances_x_y = np.append(self.x_cord-x_array, self.y_cord-y_array).reshape(2,-1)
        # print(f'distances_x_y: {distances_x_y}')
        self.distance_to_nodes = np.linalg.norm(distances_x_y, axis=0)
        self.ordered_nodes_by_distance = np.argsort(self.distance_to_nodes)

    def distance_x(self, other_node):
        x_distance = self.x_cord - other_node.x_cord
        return x_distance

    def distance_y(self, other_node):
        y_distance = self.y_cord - other_node.y_cord
        return y_distance