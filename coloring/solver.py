#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import networkx as nx
import matplotlib.pyplot as plt
import random
import colorsys
import numpy as np
from Nodes import Nodes
import time


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    print('Node count: ', node_count, ', edge_count: ', edge_count)

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    for i in range(node_count):  # genero los nodos
        Nodes(i)
    for i in edges:
        Nodes.nodes_list[i[0]].add_connection(i[1])
        Nodes.nodes_list[i[1]].add_connection(i[0])

    print('Nodes in problems: ', [i.node_id for i in Nodes.nodes_list])

    # Solucion Greedy
    for i in Nodes.nodes_list:
        if len(i.available_colours) == 0:
            Nodes.colour_base.append(Nodes.colour_base[-1]+1)
            for j in Nodes.nodes_list:
                j.add_available_colour(Nodes.colour_base[-1])
            # print('base: ', Nodes.colour_base)
            i.set_colour(i.available_colours[0])
        else:
            i.set_colour(i.available_colours[0])

    # # Plot de la solucion
    # N = len(Nodes.colour_base)
    # HSV_tuples = [(x * 1.0 / N, 1, 1) for x in range(N)]
    # RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    # #random.shuffle(RGB_tuples,)
    #
    # G = nx.DiGraph()
    # G.add_edges_from(edges)
    # colours_to_plot = [RGB_tuples[Nodes.nodes_list[i].colour] for i in G.nodes] #G.nodes lista de nodos a la hora de graficar
    # pos = nx.spring_layout(G,seed=42,k=2)
    # nx.draw_networkx_nodes(G, pos, node_size=500, node_color= colours_to_plot)
    # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black',arrowstyle='-')
    # nx.draw_networkx_labels(G, pos)
    # plt.show()
    # print('colours: ',[i.colour for i in Nodes.nodes_list])

    n_greedy = len(Nodes.colour_base)
    print('n_greedy: ', n_greedy)

    # Verifico factibilidad con menos colores

    def make_restriction(decision_order): # Me falta reiniciar los colours used!!
        if len(decision_order)>0:
            node_id = decision_order[-1][0]
            colour = decision_order[-1][1]
            number_available =  decision_order[-1][2]
            # Tengo que aplicar la restriccion y borrar la restricciones del que no pude decidir.
            if number_available >= 2:
                Nodes.nodes_list[node_id].add_decision_restriction(colour)
                Nodes.nodes_list[node_id].restore_colour()
                del decision_order[-1]
                return True
                # para optimizar tendria que sacar color,  re-calcular propagaciones de nodos que esta en contacto
            elif number_available == 1:
                Nodes.nodes_list[node_id].reset_decision_restrictions()
                Nodes.nodes_list[node_id].restore_colour()
                del decision_order[-1]
                return make_restriction(decision_order)
            #return True
        return False

    break_for = False
    break_while = False
    optimal = 0
    n_used = n_greedy
    for n_colours in range(n_greedy, 1, -1):  #Planteo un color a probar
        if n_used < n_colours:
            continue
        start_time = time.time()
        count = 0
        print(f'probando con {n_colours} colores')
        Nodes.colour_base = list(range(n_colours))  # Fijo la cantidad de colores en la base de la clase
        print(Nodes.colour_base)
        Nodes.colours_used = []
        for i in Nodes.nodes_list:
            i.reset_colour()
            i.set_available_as_base()
            i.reset_decision_restrictions()
        decision_order = []
        while not all([(lambda x: x != None)(i.colour) for i in Nodes.nodes_list]):  # Hasta que todos los nodos no esten coloreados
            if time.time() - start_time > 120:
                break_while = True
                break_for = True
                print('break activated')
                break
            #print('decision order: ', decision_order)
            Nodes.calc_df_nd()  #re-calculo las listas
            df_order = np.lexsort((Nodes.nodes_n_connections_nd, Nodes.nodes_df, Nodes.nodes_painted)) # id ordenados por menos df
            id_decide = df_order[0]
            #print('id decide: ', id_decide)
            #print(Nodes.nodes_list[id_decide].possible_colours())
            successful_set_possible_colour, number_available = Nodes.nodes_list[id_decide].set_possible_colour()
            #print('successful_set_possible_colour: ', successful_set_possible_colour)
            if not successful_set_possible_colour:  # Ejecuta si se quedo sin colores para decidir
                successful_make_restriction = make_restriction(decision_order)
                #print('restrictions: ', [(i.node_id, i.decision_restrictions) for i in Nodes.nodes_list if len(i.decision_restrictions)>0])
                if not successful_make_restriction:
                    break_while = True
                    break_for = True
                    print('break activated')
                    optimal = 1
                    break
                #Nodes.nodes_list[id_decide].reset_decision_restrictions()
                #decision_order = []  # Reinicio colores, decisiones y bases, NO restricciones por decision.
                #Nodes.colours_used = []
                #for i in Nodes.nodes_list:
                #    i.reset_colour()
                #    i.set_available_as_base()
                #print('restriction implemented', [(i.node_id, i.decision_restrictions) for i in Nodes.nodes_list if len(i.decision_restrictions)>0])
                continue
            else:
                decision_order.append((id_decide, Nodes.nodes_list[id_decide].colour, number_available))
                #print('decisions: ', decision_order,'\nRestrictions: ', [(i.node_id, i.decision_restrictions) for i in Nodes.nodes_list if len(i.decision_restrictions)>0], '\n', end='\r')
            if break_while:
                print(n_colours,'no es factible, break activated')
                break_for = True# Hacer un break que salga del for
                break
        if not break_for:
            n_used = len(Nodes.colours_used)
            output_data_0 = str(n_used)# + ' ' + str(optimal) + '\n'
            output_data_1 = ' '.join(map(str, [node.colour for node in Nodes.nodes_list]))
        if break_for:
            break


    output_data = output_data_0 + ' ' + str(optimal) + '\n' + output_data_1

        # #N = n_colours
        # HSV_tuples = [(x * 1.0 / N, 1, 1) for x in range(N)]
        # RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
        # #random.shuffle(RGB_tuples)
        #
        # G = nx.DiGraph()
        # G.add_edges_from(edges)
        # colours_to_plot = [RGB_tuples[Nodes.nodes_list[i].colour] for i in G.nodes]  # G.nodes lista de nodos a la hora de graficar
        # pos = nx.spring_layout(G, seed=42, k=2)
        # nx.draw_networkx_nodes(G, pos, node_size=500, node_color=colours_to_plot)
        # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrowstyle='-')
        # nx.draw_networkx_labels(G, pos)
        # plt.title(f'{n_colours}')
        # plt.show()
        # print('colours: ', [i.colour for i in Nodes.nodes_list])
        # print(n_colours, ' es factible')

            #Si no llego a una solucion, el la ultima decision pasa a ser una restriccion y tengo que resetear restricciones posteriores

    # prepare the solution in the specified output format
    #output_data = str(len(Nodes.colour_base)) + ' ' + str(0) + '\n'
    #output_data += ' '.join(map(str, [node.colour for node in Nodes.nodes_list]))
    print('doing dell')
    del Nodes.nodes_list[:]
    Nodes.nodes_list = []
    Nodes.nodes_painted = []
    Nodes.colour_base = [0]
    Nodes.colours_used= []
    Nodes.nodes_df = []
    Nodes.nodes_n_connections_nd = []

    return output_data


# if __name__ == '__main__':
#     import sys
#     file_location = None
#
#     if len(sys.argv) <= 1:
#         a = input('introducir a: ')
#         b = input('introducir b: ')
#         file_location = f'data/gc_{a}_{b}'
#
#     if len(sys.argv) > 1 or file_location is not None:
#         if file_location is None:
#             file_location = sys.argv[1].strip()
#         with open(file_location, 'r') as input_data_file:
#             input_data = input_data_file.read()
#         print(solve_it(input_data))
#     else:
#         print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
