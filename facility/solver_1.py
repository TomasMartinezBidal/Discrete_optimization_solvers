#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial as sc

Point = namedtuple("Point", ['x', 'y'])  # Creo objetos namedtupple, son tuplas que se pueden acceder por nombre.
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


from plot_solution_file import plot_solution
# def plot_solution(facilities, customers, solution=[]):
#
#     fig, ax = plt.subplots(nrows=1,ncols=1)
#     ax.scatter(facilities.x, facilities.y)
#     ax.scatter(customers.x, customers.y)
#     for index, row in facilities.iterrows():
#         plt.annotate(f'f{index}', row.loc[['x','y']])
#     for index, row in customers.iterrows():
#         plt.annotate(f'c{index}', row.loc[['x','y']])
#     if len(solution) > 0:
#         for i in range(len(solution)):
#             customer_xy = customers.loc[[i],['x', 'y']].values
#             facility_xy = facilities.loc[[solution[i]], ['x', 'y']].values
#             coords = np.concatenate([customer_xy,facility_xy])
#             ax.plot(coords[:,0], coords[:,1], c='g')
#     name = file_location.split('\\')[1]
#     fig.suptitle(f'{name}')
#     plt.savefig(f'images\\new_fig.png', dpi=1000)
#     return None



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])  # number of facilities
    customer_count = int(parts[1])  # number of customers

    facilities = pd.DataFrame(columns=['setup_cost', 'capacity', 'x', 'y'])
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.loc[i - 1] = parts
    facilities = facilities.astype({'setup_cost':float, 'capacity':int, 'x':float, 'y':float})
    customers = pd.DataFrame(columns=['demand', 'x', 'y'])
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.loc[i - 1 - facility_count] = parts
    customers = customers.astype({'demand':int, 'x':float, 'y':float})

    array_x_y = pd.concat([facilities.loc[:,['x','y']], customers.loc[:,['x','y']]])

    distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
    distances_facilities = distances[0:facility_count, 0:facility_count]
    distances_customers = distances[facility_count:, facility_count:]
    distances_facilities_customers = distances[0:facility_count, facility_count:]

    # print(distances_facilities)
    # print(distances_customers)
    # print(distances_facilities_customers)

    ordered_distances_facilities = np.argsort(distances_facilities, axis=0)
    ordered_distances_customers = np.argsort(distances_customers, axis=0)
    ordered_indexfacility_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)
    ordered_indexcustomer_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=0).T

    # Descripcion de greedy:
    # 1. Que vaya customer por customer y lo agregue a la facility mas cercana, que este abierta y tenga capacidad.
    # Agarro un customer, veo que facilities estan abiertas y con que capacidad. Las ordeno por distancias y capacidad
    # Si no tengo ninguna tengo que abrir una nueva!

    solution = np.ones(customer_count) * -1  # lista con asignacion de cada customer con -1 al inicio.
    opened_facilities = np.zeros(facility_count, dtype=bool)  # Mascara booleana con las facilities abiertas
    capacity_remaining = facilities.capacity.copy()  # lista de capacidad de cada facility
    asigned = np.zeros(customer_count)  # Mascara booleana con los customers asignados
    obj = 0

    for customer in range(customer_count):
        demand = customers.loc[customer, 'demand']
        if capacity_remaining.loc[opened_facilities].max() > demand:
            facilities_sufficient_capacity_mask = (capacity_remaining > demand) & (opened_facilities)
            ordered_facilities = ordered_indexcustomer_distances_facilities_customers[customer]
            ordered_open_facilities_w_cap = ordered_facilities[facilities_sufficient_capacity_mask[ordered_facilities]]
            closest_facility = ordered_open_facilities_w_cap[0]
            solution[customer] = int(closest_facility)
            capacity_remaining[closest_facility] -= demand
            obj += length(customers.loc[customer], facilities.loc[closest_facility])
        else:
            # Open the closest facilitie whith suficient capacity
            facilities_sufficient_capacity_mask = (capacity_remaining > demand) & (~opened_facilities)
            ordered_facilities = ordered_indexcustomer_distances_facilities_customers[customer]
            ordered_closed_facilities_w_cap = ordered_facilities[facilities_sufficient_capacity_mask[ordered_facilities]]
            new_facility_opened = ordered_closed_facilities_w_cap[0]
            opened_facilities[new_facility_opened] = True
            obj += facilities.loc[new_facility_opened, 'setup_cost']
            solution[customer] = int(new_facility_opened)
            capacity_remaining[new_facility_opened] -= demand
            obj += length(customers.loc[customer], facilities.loc[new_facility_opened])

    solution = solution.astype(int)
    plot_solution(facilities, customers, file_location, solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()  # input_data is a string at this instance.
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

