#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
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

    from hyperparameters import df_hyperparameters

    hyper_parametes = df_hyperparameters.loc[customer_count, facility_count]

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

    # Generate greedy solution on f and x
    aux.greedy_solver(customer_count, facilities, customers, ordered_rowcustomer_distances_facilities_customers, x, f,
                  greedy_facility_maks=None, plot=False)


    min_cost = np.inf
    print(f'Objective: {hyper_parametes.objective_threshold}')
    # start_time = time.time()
    i = 0

    time_limit = hyper_parametes.time_limit if hyper_parametes.solve_full else 30
    # if ~hyper_parametes.solve_full:
    while min_cost >= hyper_parametes.objective_threshold:
    # for i in tqdm(range(facility_count * 3)):
        facilities_subset = np.random.choice(facility_count)
        facilities_subset = ordered_distances_facilities[:,facilities_subset][:hyper_parametes.facilities_subset]

        customers_subset = np.arange(customer_count)[x.loc[:,facilities_subset].apply(lambda row: row.any(), axis=1)]

        for facility_id in facilities_subset:
            previous_value = f.loc[facility_id,'facility_open']
            f.loc[facility_id,'facility_open'] = pulp.LpVariable(name=f'Facility:{facility_id}', cat='Binary')
            f.loc[facility_id,'facility_open'].setInitialValue(val=previous_value)
            for customer_id in customers_subset:
                previous_value = x.loc[customer_id, facility_id]
                x.loc[customer_id, facility_id] = pulp.LpVariable(name=f'Customer:{customer_id}_,Facility:{facility_id}', cat='Binary')
                x.loc[customer_id, facility_id].setInitialValue(val=previous_value)

        prob = pulp.LpProblem(name='facility_location', sense=pulp.const.LpMinimize)

        for customer in customers_subset:  # each customer is allocated to one facility only.
            prob += pulp.lpSum(x.loc[customer]) == 1
        # print('done customers eq')

        for facility in facilities_subset:
            prob += pulp.lpSum(x.loc[:, facility] * customers.loc[:, 'demand']) <= facilities.loc[facility, 'capacity'] * f.loc[facility, 'facility_open']
        # print('done facilities eq')

        prob += pulp.lpSum((distances_facilities_customers * x).values) + pulp.lpSum(
            (facilities.loc[:, 'setup_cost'] * f.loc[:, 'facility_open'].T).values)

        prob.solve(pulp.PULP_CBC_CMD(msg=0, warmStart=True, timeLimit=time_limit, ))

        for facility_id in facilities_subset:
            f.loc[facility_id, 'facility_open'] = int(pulp.value(f.loc[facility_id, 'facility_open']))
            for customer_id in customers_subset:
                x.loc[customer_id, facility_id] = int(pulp.value(x.loc[customer_id, facility_id]))

        if pulp.value(prob.objective) < min_cost:
            min_cost = pulp.value(prob.objective)
            print(min_cost)
            print(i)
            # min_x = x.copy()
            # min_f = f.copy()
        i += 1
        if hyper_parametes.solve_full:
            break

    solution = list(x.values.argsort()[:, -1])
    output_data = '%.2f' % min_cost + ' ' + str(0) + '\n'
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
