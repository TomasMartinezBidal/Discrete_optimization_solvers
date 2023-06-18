#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mip

import scipy.spatial as sc
# from plot_solution_file import plot_solution


Point = namedtuple("Point", ['x', 'y'])  # Creo objetos namedtupple, son tuplas que se pueden acceder por nombre.
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])  # number of facilities
    print(facility_count)
    customer_count = int(parts[1])  # number of customers
    print(customer_count)

    facilities = pd.DataFrame(columns=['setup_cost', 'capacity', 'x', 'y'])
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.loc[i - 1] = parts
    facilities = facilities.astype({'setup_cost':float, 'capacity':int, 'x':float, 'y':float})
    print('done facilities')
    customers = pd.DataFrame(columns=['demand', 'x', 'y'])
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.loc[i - 1 - facility_count] = parts
    customers = customers.astype({'demand':int, 'x':float, 'y':float})
    print('done customers')

    array_x_y = pd.concat([facilities.loc[:,['x','y']], customers.loc[:,['x','y']]])

    distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
    # distances_facilities = distances[0:facility_count, 0:facility_count]
    # distances_customers = distances[facility_count:, facility_count:]
    distances_facilities_customers = distances[facility_count:, :facility_count]

    # print(distances_facilities)
    # print(distances_customers)
    # print(distances_facilities_customers)

    # ordered_distances_facilities = np.argsort(distances_facilities, axis=0)
    # ordered_distances_customers = np.argsort(distances_customers, axis=0)
    # ordered_indexfacility_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)
    ordered_indexcustomer_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=0).T

    mip_model = mip.Model()
    # x = [mip_model.add_var(var_type=mip.BINARY) for i in range(customer_count) for j in range(facility_count)]  # customer-facility boolean variable
    x = pd.DataFrame(np.array([mip_model.add_var(var_type=mip.BINARY) for _ in range(customer_count * facility_count)])\
                     .reshape((customer_count, facility_count)))
    # f = [mip_model.add_var(var_type=mip.BINARY) for i in range(facility_count)]  # Facility open boolean var
    f = pd.DataFrame(np.array([mip_model.add_var(var_type=mip.BINARY) for _ in range(facility_count)]))
    demands = list(customers.loc[:, 'demand']) * facility_count
    for customer in range(customer_count):  # each customer is allocated to one facility only.
        mip_model += x.loc[customer].sum() == 1
        # mip_model += mip.xsum(x[j] for j in range(customer * facility_count, (customer + 1) * facility_count, 1)) == 1
    print('done customers eq')
    for facility in range(facility_count):

        mip_model += (x.loc[:,facility] * customers.loc[:, 'demand']).sum() <= (facilities.loc[facility, 'capacity'] * f.loc[facility]).values[0]
        # mip_model += mip.xsum(x[j] * demands[j] for j in range(facility, facility_count * customer_count, facility_count)) \
        #              <= facilities.loc[facility, 'capacity'] * f[facility]
    print('done facilities eq')

    distances_flatten = distances_facilities_customers.T.flatten()

    # mip_model.objective = mip.xsum(distances_flatten[j] * x[j] for j in range(facility_count * customer_count)) + \
    #                       mip.xsum(facilities.loc[facility, 'setup_cost'] * f[facility] for
    #                                facility in range(facility_count))

    # var_1 = (distances_facilities_customers * x).values.sum()
    # var_2 = (facilities.loc[:,'setup_cost'] * f.T).values.sum()
    print('loading objective function')
    mip_model.objective = (distances_facilities_customers * x).values.sum() + (facilities.loc[:,'setup_cost'] * f.T).values.sum()
    print('done loading objective function')



    mip_model.max_gap = 0.05
    print('start solving')
    status = mip_model.optimize(max_seconds=600)
    print(status)
    decition_variables_array = np.array([i.x for i in x.values.flatten()])
    # print(bool(sum(x[j].x for j in range(faciliy, facility_count * customer_count, customer_count))) for faciliy in range(facility_count))
    decition_variables_array = decition_variables_array.reshape((customer_count, facility_count))
    solution = list(decition_variables_array.argsort()[:, -1])
    # plot_solution(facilities, customers, file_location, solution)


    obj = mip_model.objective_value

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

    input_file_name = input('Ingresar un nombre de problema, si se ingresa \'0\' se utilizara el problema fl_3_1: ')

    file_location = ''

    if input_file_name == '0':
        file_location = 'data\\fl_3_1'

    elif len(input_file_name) > 0:
        file_location = 'data\\' + input_file_name

    if file_location:
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()  # input_data is a string at this instance.
        print(solve_it(input_data))

    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

