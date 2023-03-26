#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import time
from collections import namedtuple
import pandas as pd
import numpy as np
import random
from funciones_tsp import dist_path, nearest_points
from nodos import Nodes, fun_list_calc_ordered_nodes_by_distance, fun_list_calc_ordered_nodes_by_distance_2
import math
from solution import Solution

import matplotlib.pyplot as plt

from multiprocessing import Process, cpu_count
import concurrent.futures
import itertools


Point = namedtuple("Point", ['x', 'y'])

def solve_it(input_data):

    Nodes.nodes_list = []
    Nodes.y_coords = []
    Nodes.x_coords = []

    Solution.solution_list = []
    Solution.solution_distances = []

    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    print(f'Node count: {nodeCount}')

    x_y_array = np.array([lines[i].split() for i in range(1, nodeCount+1)])
    x_y_array = x_y_array.astype(float)

    print('----------------generating node objects---------------')

    Nodes.nodes_list = []
    Nodes.total_nodes = 0
    Nodes_class = Nodes

    for i in range(nodeCount):
        Nodes_class(x_y_array[i,0],x_y_array[i,1])

    print('len nodes list: ', len(Nodes_class.nodes_list))

    print('----------------Objects created----------------')

    time_create_objects_start = time.time()
    flag = False  # Decide if calculate distances with multiprocessing or not
    if flag:
        available_cpus = cpu_count()  # Number of cores available
        available_cpus = 2  # 2 if available_cpus <= 4 else available_cpus - 2  # Leave two cores for other processes
        print(f'available cpus: {available_cpus}')
        nodes_per_chunk = nodeCount/available_cpus  # Nodes i want to precess in each core
        indexes = np.floor(nodes_per_chunk*np.arange(available_cpus+1)).astype(int)  # Divide nodes in qcut fashion
        start_index = indexes[:-1]  # Make list of indexes for node lists, start and end indexes
        end_index = indexes[1:]

        # for i,j in zip(start_index, end_index):
        #     print(i, j)

        chunks = [Nodes_class.nodes_list[i:j] for i,j in zip(start_index, end_index)]  # List of list of nodes for multiprocesing

        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            processes = executor.map(fun_list_calc_ordered_nodes_by_distance, chunks, [Nodes.x_coords]*len(chunks), [Nodes.y_coords]*len(chunks))

        new_nodes = list(itertools.chain(*processes))

        print('printing new nodes')

        # for i in new_nodes:
        #     print(i)
        #     print(i.ordered_nodes_by_distance)

        Nodes_class.nodes_list = new_nodes
    else:
        fun_list_calc_ordered_nodes_by_distance_2(Nodes_class)
        # fun_list_calc_ordered_nodes_by_distance(Nodes_class.nodes_list)

    # #Generamos una solucion greedy pero de un camino cerrado

    time_create_objects_end = time.time()

    print(f'elapsed time in calculating distances: {(time_create_objects_end-time_create_objects_start)} seg')

    print('----------------starting solution----------------')

    time_create_first_solution_start = time.time()

    neighbour_count = min(math.ceil(nodeCount**0.5), 3)
    print(f'Using {neighbour_count} neighbours')

    first_solution = Solution(nodeCount, Nodes, neighbour_count)
    first_solution.add_zero_to_path()

    previous_progress = 0
    image_counter = 0

    while len(first_solution.not_visited_index) > 0:
        #print('first_solution_n\n',first_solution.neighbours)
        first_solution.add_node_to_path_2(number_nodes_check_when_adding = 3)
        progress = np.floor((len(first_solution.path_index) - 1) / nodeCount*100)
        if previous_progress < progress:
            print('\r',f'Solution progress: {progress}%', end='')
            previous_progress = progress

    time_create_first_solution_end = time.time()

    print(f'\nelapsed time in calculating first solution: {(time_create_first_solution_end-time_create_first_solution_start)} seg')


    print("Generated first solution")

    # first_solution.plot_solution(x_y_array,'first_solution')

    print("Generating two opt")

    first_solution.two_opt()

    # first_solution.plot_solution(x_y_array,'two_opt')

    print('two opt finished')

    print('generating swaps')

    first_solution.swap_node_to_between()

    # first_solution.plot_solution(x_y_array,'first_swap')

    print('swaps finished')

    print('generating more swaps')

    first_solution.swap_node_to_between()

    # first_solution.plot_solution(x_y_array,'second_swap')

    print('finished more swaps')

    print('some k_opt')

    # first_solution.plot_solution(x_y_array, 'pre_k_opt')
    # time.sleep(2)

    time_init = time.time()
    previous_progress = 0
    k_opt_iterations = nodeCount*10 if nodeCount < 30000 else nodeCount

    if nodeCount < 30000:
        for opt_counter in range(k_opt_iterations):
            progress = np.floor((opt_counter) / (k_opt_iterations - 1) * 100)
            if previous_progress < progress:
                print('\r',f'k-opt progress: {progress}%', end='')
                previous_progress = progress
            t2_node_id = random.randint(0, nodeCount-1)  # Elijo un nodo al azar
            t1_node_id = first_solution.adjacent[t2_node_id][np.random.randint(0,2)]  # node to the right of t2, why not left??
            first_solution.k_opt(t2_node_id, t1_node_id, k=6, x_y_array=x_y_array)  # Aplico k-opts
            # first_solution.plot_solution(x_y_array, f'opt_{opt_counter}_{round(first_solution.distance, 2)}')
            first_solution = Solution.solution_list[0]

    print('')

    time_end = time.time()

    print(f'elapsed time in k opt: {(time_end-time_init)} seg')

    print('finished some k_opt')

    print('final two opt')

    Solution.solution_list[0].two_opt()

    # Solution.solution_list[0].plot_solution(x_y_array,'final_two_opt')

    print('two opt finished')

    print('distancia optimizada: ', Solution.solution_list[0].distance)

    # prepare the solution in the specified output format
    output_data = '%.2f' % Solution.solution_list[0].distance + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, Solution.solution_list[0].path_index[:-1]))

    return output_data

# if __name__ == '__main__':
#     file_location = "data/tsp_70_1"
#     with open(file_location, 'r') as input_data_file:
#         input_data = input_data_file.read()
#     print(solve_it(input_data))

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        # file_location = 'data/tsp_5934_1'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
        print('can still do stuff')
        del Solution
        del Nodes
        del Point
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver0.py ./data/tsp_51_1)')

