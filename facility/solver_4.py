#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

import pulp

import scipy.spatial as sc
from plot_solution_file import plot_solution


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

    # Genero df con mis facilities y mnis customers
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
    distances_facilities = distances[0:facility_count, 0:facility_count]
    distances_customers = distances[facility_count:, facility_count:]
    distances_facilities_customers = distances[facility_count:, :facility_count]  # Customer in row, facility in column

    ordered_distances_facilities = np.argsort(distances_facilities, axis=0)
    ordered_distances_customers = np.argsort(distances_customers, axis=0)
    # ordered_rowfacility_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)
    ordered_rowcustomer_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)

    x = pd.DataFrame(np.zeros((customer_count, facility_count), dtype='int'))  # df para las decition variables

    f = pd.DataFrame(np.zeros(facility_count, dtype='int'), columns=['facility_open'])
    f = f.join(facilities.loc[:,'capacity'])
    f.facility_open = 1

    for customer in range(customer_count):
        open_facilities = f.facility_open
        open_facilities_w_capacity_maks = (f.capacity > customers.loc[customer,'demand']) & open_facilities
        open_facilities_w_capacity_dist_ordered = ordered_rowcustomer_distances_facilities_customers[customer][open_facilities_w_capacity_maks[ordered_rowcustomer_distances_facilities_customers[customer]]]
        facility_to_be_asigned = open_facilities_w_capacity_dist_ordered[0]
        f.loc[facility_to_be_asigned, 'capacity'] -= customers.loc[customer, 'demand']
        x.loc[customer, facility_to_be_asigned] = 1

    # Close all facilities that are not occupied.
    mask_close_facilities = f.capacity == facilities.capacity
    f.loc[mask_close_facilities, 'facility_open'] = 0

    hue_array = (~mask_close_facilities).map({True:'Open', False:'Close'})
    facility_mean_capacity = facilities.loc[:,'capacity'].mean()
    facility_mean_setup_cost =  facilities.loc[:,'setup_cost'].mean()
    if facility_count <= 100 & customer_count <= 1000:
        facility_mean_capacity = 0
        facility_mean_setup_cost = 0
    extra_facilities_mask = (facilities.loc[:,'capacity'] >= facility_mean_capacity) & (facilities.loc[:,'setup_cost'] <= facility_mean_setup_cost)
    extra_facilities_mask = extra_facilities_mask & mask_close_facilities
    hue_array[extra_facilities_mask] = 'New_open'
    # g = sns.jointplot(data=facilities, x='setup_cost', y='capacity', hue=hue_array)
    # plt.show()

    mask_open_facilities_pulp = ~mask_close_facilities | extra_facilities_mask

    # decition_variables_array = np.array([ i if isinstance(i,np.int32) else i.varValue for i in x.values.flatten()])
    # decition_variables_array = decition_variables_array.reshape((customer_count, facility_count))
    solution = list(x.values.argsort()[:, -1])
    print(f'Greedy solution:\n{solution}')
    plot_solution(facilities=facilities, customers=customers, solution=solution, extra='greedy')

    if customer_count >= 2000:
        obj = facilities.loc[~mask_close_facilities, 'setup_cost'].sum() + distances_facilities_customers[x.astype(bool)].sum()

        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))
        return output_data

    # Paso a armar el solver con pulp

    prob = pulp.LpProblem(name='facility_location',  sense=pulp.const.LpMinimize)

    available_facilities = np.array(range(facility_count))[mask_open_facilities_pulp]

    for i in tqdm(range(customer_count)):
        for j in available_facilities:
            value = x.loc[i,j]
            x.loc[i,j] = pulp.LpVariable(name=f'Customer:{i}_,Facility:{j}', cat='Binary') #  mip_model.add_var(var_type=mip.BINARY)
            x.loc[i, j].setInitialValue(val=value)
    print('done setting customers')

    f.loc[mask_open_facilities_pulp,'facility_open'] = np.array([pulp.LpVariable(name=f'Facility:{i}', cat='Binary') for i in available_facilities])
    for i in f.loc[~mask_close_facilities,'facility_open']:
        i.setInitialValue(val=1)
    print('done setting facilities')

    for customer in tqdm(range(customer_count)):  # each customer is allocated to one facility only.
        prob += pulp.lpSum(x.loc[customer]) == 1
    print('done customers eq')

    for facility in tqdm(range(facility_count)):
        prob += pulp.lpSum(x.loc[:, facility] * customers.loc[:, 'demand']) <= facilities.loc[facility, 'capacity'] * f.loc[facility, 'facility_open']
    print('done facilities eq')

    prob += pulp.lpSum((distances_facilities_customers * x).values) + pulp.lpSum((facilities.loc[:,'setup_cost'] * f.loc[:, 'facility_open'].T).values)

    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*60, warmStart=True))#, warmStart=True)

    status = pulp.LpStatus[prob.status]
    print(status)

    decition_variables_array = np.array([i if isinstance(i, int) else i.varValue for i in x.values.flatten()])
    decition_variables_array = decition_variables_array.reshape((customer_count, facility_count))
    solution = list(decition_variables_array.argsort()[:, -1])

    plot_solution(facilities=facilities, customers=customers, solution=solution, extra='MIP')

    obj = pulp.value(prob.objective)  # mip_model.objective_value

    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))


    # Linea para cuando no uso pulp
    # obj = facilities.loc[~mask_close_facilities, 'setup_cost'].sum() + distances_facilities_customers[x.astype(bool)].sum()
    #
    # output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    # output_data += ' '.join(map(str, solution))

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
# if __name__ == '__main__':
#     import sys
#
#     if len(sys.argv) > 1:
#         file_location = sys.argv[1].strip()
#         with open(file_location, 'r') as input_data_file:
#             input_data = input_data_file.read()  # input_data is a string at this instance.
#         print(solve_it(input_data))
#
#     input_file_name = input('Ingresar un nombre de problema, si se ingresa \'0\' se utilizara el problema fl_3_1: ')
#
#     file_location = ''
#
#     if input_file_name == '0':
#         file_location = 'data\\fl_3_1'
#
#     elif len(input_file_name) > 0:
#         file_location = 'data\\' + input_file_name
#
#     if file_location:
#         with open(file_location, 'r') as input_data_file:
#             input_data = input_data_file.read()  # input_data is a string at this instance.
#         print(solve_it(input_data))
#
#     else:
#         print(
#             'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
#
