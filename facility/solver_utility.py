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

    print(f'Facility count: {facility_count}, customer count: {customer_count}')
    return 'No solution'

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
