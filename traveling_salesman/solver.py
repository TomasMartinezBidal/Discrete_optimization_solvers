#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import time
from collections import namedtuple
import pandas as pd
import numpy as np
import random
from funciones_tsp import dist_path, nearest_points
from nodos import Nodes
import math
from solution import Solution

import matplotlib.pyplot as plt


Point = namedtuple("Point", ['x', 'y'])


def solve_it(input_data):

    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    print(f'Node count: {nodeCount}')

    x_y_array = np.array([lines[i].split() for i in range(1, nodeCount+1)])
    x_y_array = x_y_array.astype(float)

    print('----------------generating node objects---------------')

    Nodes.nodes_list = []
    Nodes.total_nodes = 0

    for i in range(nodeCount):
        Nodes(x_y_array[i,0],x_y_array[i,1])

    print('----------------Objects created----------------')

    previous_progress = 0

    for i in Nodes.nodes_list:
        i.calc_ordered_nodes_by_distance()

        progress = np.floor((i.node_id) / (nodeCount-1) * 100)
        if previous_progress < progress:
            print('\r', f'Distance calculation progress: {progress}%', end='')
            previous_progress = progress
    print('')
    #Generamos una solucion greedy pero de un camino cerrado

    print('----------------starting solution----------------')


    neighbour_count = min(math.ceil(nodeCount**0.5), 30)
    print(f'Using {neighbour_count} neighbours')

    first_solution = Solution(nodeCount, Nodes, neighbour_count)
    first_solution.add_zero_to_path()

    previous_progress = 0
    image_counter = 0
    #fig, ax = plt.subplots(figsize=(10, 10))
    while len(first_solution.not_visited_index) > 0:
        #print('first_solution_n\n',first_solution.neighbours)
        first_solution.add_node_to_path()
        progress = np.floor((len(first_solution.path_index) - 1) / nodeCount*100)
        if previous_progress < progress:
            print('\r',f'Solution progress: {progress}%', end='')
            previous_progress = progress
        # # plt.ion()
        # ax.plot(x_y_array[first_solution.path_index, 0], x_y_array[first_solution.path_index, 1])
        # ax.scatter(x_y_array[:, 0], x_y_array[:, 1],s=10)
        # fig.savefig(f'images\{image_counter}.png',dpi=200)
        # ax.cla()
        # image_counter += 1
        # # plt.draw()
        # # plt.show()
    print("\nGenerated first solution")

    first_solution.plot_solution(x_y_array,'first_solution')

    print("Generating two opt")

    first_solution.two_opt()

    first_solution.plot_solution(x_y_array,'two_opt')

    print('two opt finished')

    print('generating swaps')

    first_solution.swap_node_to_between()

    first_solution.plot_solution(x_y_array,'first_swap')

    print('swaps finished')

    print('generating more swaps')

    first_solution.swap_node_to_between()

    first_solution.plot_solution(x_y_array,'second_swap')

    print('finished more swaps')

    print('final two opt')

    first_solution.two_opt()

    first_solution.plot_solution(x_y_array,'final_two_opt')

    print('two opt finished')

    print('distancia optimizada: ', first_solution.distance)

    # prepare the solution in the specified output format
    output_data = '%.2f' % first_solution.distance + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, first_solution.path_index[:-1]))

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
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver0.py ./data/tsp_51_1)')

