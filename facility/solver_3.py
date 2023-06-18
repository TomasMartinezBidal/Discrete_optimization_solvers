#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pulp

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
    distances_facilities_customers = distances[facility_count:, :facility_count]  # Customer in row,facility in column

    # print(distances_facilities)
    # print(distances_customers)
    # print(distances_facilities_customers)

    # ordered_distances_facilities = np.argsort(distances_facilities, axis=0)
    # ordered_distances_customers = np.argsort(distances_customers, axis=0)
    # ordered_indexfacility_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)
    ordered_indexcustomer_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)

    prob = pulp.LpProblem(name='facility_location',  sense=pulp.const.LpMinimize)

    x = pd.DataFrame(np.zeros((customer_count, facility_count)))

    facility_proximity_factor = 1  # To be used when analyzing how many facilities aew available to a customer.
    print(f"facility proximity factor: {facility_proximity_factor}")
    proximity_number_facilities = np.ceil(facility_proximity_factor * facility_count).astype(int)


    for i in tqdm(range(customer_count)):
        for j in range(facility_count):
            x.loc[i,j] = pulp.LpVariable(name=f'Customer:{i}_,Facility:{j}', cat='Binary') #  mip_model.add_var(var_type=mip.BINARY)

    # ordered_indexcustomer_distances_facilities_customers[i,j]

    f = pd.DataFrame(np.array([pulp.LpVariable(name=f'Facility:{i}', cat='Binary') for i in range(facility_count)]))

    for customer in tqdm(range(customer_count)):  # each customer is allocated to one facility only.
        prob += pulp.lpSum(x.loc[customer]) == 1

    print('done customers eq')
    for facility in tqdm(range(facility_count)):
        prob += pulp.lpSum(x.loc[:, facility] * customers.loc[:, 'demand']) <= facilities.loc[facility, 'capacity'] * f.loc[facility, 0]

    print('done facilities eq')

    add_third_restriction = True

    if add_third_restriction:
        for customer in tqdm(range(customer_count)):
            for facility in range(facility_count):
                prob += x.loc[customer, facility] <= f.loc[facility, 0]

    add_forth_restriction = False
    if add_forth_restriction:
        for facility in tqdm(range(facility_count)):
            prob += pulp.lpSum(x.loc[:, facility])/customer_count <= f.loc[facility, 0]

    print('done third eq')

    print('loading objective function')
    prob += pulp.lpSum((distances_facilities_customers * x).values) + pulp.lpSum((facilities.loc[:,'setup_cost'] * f.T).values)
    # mip_model.objective = (distances_facilities_customers * x).values.sum() + (facilities.loc[:,'setup_cost'] * f.T).values.sum()
    print('done loading objective function')



    # mip_model.max_gap = 0.05
    print('start solving')
    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*10))
    # status = mip_model.optimize(max_seconds=600)
    status = pulp.LpStatus[prob.status]
    print(status)

    decition_variables_array = np.array([ i if isinstance(i,float) else i.varValue for i in x.values.flatten()])
    # print(bool(sum(x[j].x for j in range(faciliy, facility_count * customer_count, customer_count))) for faciliy in range(facility_count))
    decition_variables_array = decition_variables_array.reshape((customer_count, facility_count))
    solution = list(decition_variables_array.argsort()[:, -1])
    # plot_solution(facilities, customers, file_location, solution)

    obj = pulp.value(prob.objective) # mip_model.objective_value

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

    input_file_name = input('Ingresar un nombre de problema, si se ingresa un numero de 1 a 8 se resolver ese problema del submit: ')

    file_location = ''

    files_dict = dict(zip(range(1,9), ['./data/fl_25_2', './data/fl_50_6', './data/fl_100_7', './data/fl_100_1', './data/fl_200_7', './data/fl_500_7', './data/fl_1000_2', './data/fl_2000_2']))

    # if input_file_name == '0':
    #     file_location = 'data\\fl_3_1'

    try:
        if int(input_file_name) in range(1,9):
            file_location = files_dict[int(input_file_name)]
            file_location = 'data\\' + file_location.split('/')[-1]
    except:
        pass

    # elif len(input_file_name) > 0:
    #     file_location = 'data\\' + input_file_name

    if file_location:
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()  # input_data is a string at this instance.
        print(solve_it(input_data))

    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

