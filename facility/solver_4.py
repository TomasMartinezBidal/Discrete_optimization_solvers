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

import auxiliaries as aux


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

    # Genero df con mis facilities y mis customers
    facilities, customers = aux.make_facilities_customers_df(lines=lines, facility_count=facility_count, customer_count=customer_count)

    # Generate distance arrays
    distances, distances_facilities, distances_customers, distances_facilities_customers, ordered_distances_facilities,\
        ordered_distances_customers, ordered_rowfacility_distances_facilities_customers,\
        ordered_rowcustomer_distances_facilities_customers = aux.calc_distances(facility_count=facility_count,
                                                                                customer_count=customer_count,
                                                                                facilities=facilities,
                                                                                customers=customers)

    # Generate decision variable dfs
    f, x = aux.gen_decition_df(facility_count=facility_count, customer_count=customer_count, facilities=facilities)
    # Might need correction

    # Generate efficiency mask
    # available_facilities, greedy_facility_maks = aux.efficiency_mask_generator(facility_count, facilities, customers, ordered_rowfacility_distances_facilities_customers, distances_facilities_customers)

    greedy = False
    if greedy:
        greedy_facility_maks = pd.Series(np.ones(facility_count))
        _, _, greedy_facility_maks = aux.greedy_solver(customer_count, facilities, customers,
                      ordered_rowcustomer_distances_facilities_customers, x, f, greedy_facility_maks)

        mask_open_facilities_pulp = greedy_facility_maks
        mask_open_facilities = mask_open_facilities_pulp

        solution = list(x.values.argsort()[:, -1])
        print(f'Greedy solution:\n{solution}')
        plot_solution(facilities=facilities, customers=customers, solution=solution, extra='greedy')
    else:
        greedy_facility_maks = np.ones(shape=facility_count, dtype=bool)
        mask_open_facilities_pulp = greedy_facility_maks
        mask_open_facilities = mask_open_facilities_pulp

    if customer_count >= 200000:
        obj = facilities.loc[mask_open_facilities, 'setup_cost'].sum() + distances_facilities_customers[x.astype(bool)].sum()

        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))
        return output_data

    # Paso a armar el solver con pulp

    prob = pulp.LpProblem(name='facility_location',  sense=pulp.const.LpMinimize)

    available_facilities = np.array(range(facility_count))[mask_open_facilities_pulp.astype(bool)]

    # Transform customers integer greedy solution to pulp variables
    total_facilities_available = len(available_facilities)
    for i in tqdm(range(customer_count)):
        # calculate the nearest facilities to i:
        order_indexes = available_facilities[np.argsort(np.argsort(ordered_rowcustomer_distances_facilities_customers[i])[available_facilities])]
        k1, k2 = 0.05, 50
        number_facilities_available_i_customer = np.floor(total_facilities_available*k1).astype(int) if np.floor(total_facilities_available*k1).astype(int) > k2 else k2
        facilities_available_i_customer = order_indexes[:number_facilities_available_i_customer]
        for j in facilities_available_i_customer:
        # for j in available_facilities:
                value = x.loc[i,j]
                x.loc[i,j] = pulp.LpVariable(name=f'Customer:{i}_,Facility:{j}', cat='Binary') #  mip_model.add_var(var_type=mip.BINARY)
                x.loc[i, j].setInitialValue(val=value)
    print('done setting customers')

    # Transform facility integer greedy solution to pulp variables
    f.loc[mask_open_facilities_pulp,'facility_open'] = np.array([pulp.LpVariable(name=f'Facility:{i}', cat='Binary') for i in available_facilities])
    for i in f.loc[mask_open_facilities,'facility_open']:
        i.setInitialValue(val=1)
    print('done setting facilities')

    facilities_ef_mask = greedy_facility_maks
    mask_efficient_not_greedy = ~mask_open_facilities_pulp & facilities_ef_mask
    efficient_not_greedy_facilities = pd.Series(range(facility_count))[mask_efficient_not_greedy].values
    f.loc[mask_efficient_not_greedy, 'facility_open'] = np.array([pulp.LpVariable(name=f'Facility:{i}', cat='Binary') for i in efficient_not_greedy_facilities])

    for customer in tqdm(range(customer_count)):  # each customer is allocated to one facility only.
        prob += pulp.lpSum(x.loc[customer]) == 1
    print('done customers eq')

    for facility in tqdm(range(facility_count)):
        prob += pulp.lpSum(x.loc[:, facility] * customers.loc[:, 'demand']) <= facilities.loc[facility, 'capacity'] * f.loc[facility, 'facility_open']
    print('done facilities eq')

    add_third_restriction = True

    if add_third_restriction:
        for customer in tqdm(range(customer_count)):
            for facility in available_facilities:#range(facility_count):
                prob += x.loc[customer, facility] <= f.loc[facility, 'facility_open']

    add_forth_restriction = True
    if add_forth_restriction:
        for facility in tqdm(range(facility_count)):
            prob += pulp.lpSum(x.loc[:, facility])/customer_count <= f.loc[facility, 'facility_open']

    prob += pulp.lpSum((distances_facilities_customers * x).values) + pulp.lpSum((facilities.loc[:,'setup_cost'] * f.loc[:, 'facility_open'].T).values)

    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*120, warmStart=False))#, threads=1))#, warmStart=True)

    status = pulp.LpStatus[prob.status]
    print(status)

    decition_variables_array = np.array([i if isinstance(i, int) else i.varValue for i in x.values.flatten()])
    decition_variables_array = decition_variables_array.reshape((customer_count, facility_count))
    solution = list(decition_variables_array.argsort()[:, -1])

    # plot_solution(facilities=facilities, customers=customers, solution=solution, extra='MIP')

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
