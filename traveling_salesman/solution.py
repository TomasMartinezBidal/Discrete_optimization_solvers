import time
import nodos
import numpy as np
import math
import matplotlib.pyplot as plt


class Solution:

    solution_list =[]
    solution_distances = []
    recent_k_opt_nodes_number = 10
    recent_k_opt_nodes = [-1] * recent_k_opt_nodes_number
    plot_counter = 0  # Un contador para los titulos de los graficos

    def __init__(self, node_count, nodes, neighbour_count):
        self.solution_list.append(self)
        self.node_count = node_count
        self.path_index = []
        self.neighbours = []
        self.not_visited_index = list(range(1, node_count))
        self.distance = 0
        self.neighbour_count = neighbour_count
        self.nodes = nodes
        self.adjacent = {}

    # def __del__(self):
    #     print('class deleted')

    def copy(self):
        Solution(self.node_count, self.nodes, self.neighbour_count)
        Solution.solution_list[-1].path_index = self.path_index.copy()
        Solution.solution_list[-1].not_visited_index = []
        Solution.solution_list[-1].distance = self.distance.copy()
        Solution.solution_list[-1].adjacent = self.adjacent.copy()

    @staticmethod
    def calc_solution_distances():
        Solution.solution_distances = [i.distance for i in Solution.solution_list]

    def plot_solution(self, coord_array, title='no_title'):
        plt.figure(figsize=(10, 10))
        title = f'{bin(Solution.plot_counter)}-{title}'
        plt.title(title)
        plt.plot(coord_array[self.path_index, 0], coord_array[self.path_index, 1])
        plt.scatter(coord_array[:, 0], coord_array[:, 1])
        for x, y, text in zip([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                              [self.nodes.nodes_list[i].y_cord for i in self.path_index],
                              self.path_index[:-1]):
            plt.annotate(text, (x, y + 0.1))
        plt.savefig(f'images\\{title}.png', dpi=200)
        plt.close()
        Solution.plot_counter += 1

    def add_zero_to_path(self):
        if self.path_index == []:
            self.path_index = [0, 0]
            neighbours_nodes = self.nodes.nodes_list[0].ordered_nodes_by_distance[1:self.neighbour_count + 1]  # Lista con todos los nodos
            neighbours_distance = self.nodes.nodes_list[0].distance_to_nodes[neighbours_nodes]  # lista con distancias a nodos de neighbours_nodes
            neighbours_of = np.zeros(len(neighbours_nodes))  # de cual son vecino, en esta caso es 0.
            neighbours = np.vstack((neighbours_nodes, neighbours_distance, neighbours_of))
            self.neighbours = neighbours
            self.adjacent[0] = [0, 0]

    def add_node_to_path(self, number_nodes_check_when_adding):
        node_id = int(self.neighbours[0, 0])  # El nodo mas cercano de self.neighbours[0, 0]
        mask = self.neighbours[0] == node_id  # Mascara filtrando neighbours
        solution_neighbours = self.neighbours[2, mask].astype(int)[:number_nodes_check_when_adding]  # Neighbours al nodo que quiero ingresar
        #solution_neighbours = solution_neighbours[:10] if len(solution_neighbours) > 10 else solution_neighbours
        distance_diff = np.inf

        for i in solution_neighbours:
            # print('i:',i)
            index = self.path_index.index(i)  # index of node i in the solution
            distance_right = self.nodes.nodes_list[self.path_index[index]].distance(
                self.nodes.nodes_list[self.path_index[index + 1]])
            new_distance_right = self.nodes.nodes_list[node_id].distance(
                self.nodes.nodes_list[self.path_index[index]]) + \
                                 self.nodes.nodes_list[node_id].distance(
                                     self.nodes.nodes_list[self.path_index[index + 1]])
            if index != 0:
                # print(f'i: {i}, node_id: {node_id}')
                # print(self.nodes.nodes_list[node_id].distance(self.nodes.nodes_list[self.path_index[index]]))
                # print(self.nodes.nodes_list[node_id].distance(self.nodes.nodes_list[self.path_index[index - 1]]))
                distance_left = self.nodes.nodes_list[self.path_index[index]].distance(
                    self.nodes.nodes_list[self.path_index[index - 1]])
                new_distance_left = self.nodes.nodes_list[node_id].distance(
                    self.nodes.nodes_list[self.path_index[index]]) + \
                                    self.nodes.nodes_list[node_id].distance(
                                        self.nodes.nodes_list[self.path_index[index - 1]])
            else:
                distance_left = np.inf
                new_distance_left = np.inf

            if distance_diff > (new_distance_right - distance_right):
                distance_diff = new_distance_right - distance_right
                previous_distance = distance_right
                new_distance = new_distance_right
                left = False
                best_solution_neighbour = i
                best_solution_neighbour_index = index
            if distance_diff > (new_distance_left - distance_left):
                distance_diff = new_distance_left - distance_left
                previous_distance = distance_left
                new_distance = new_distance_left
                left = True
                best_solution_neighbour = i
                best_solution_neighbour_index = index

        if left:
            other_neighbour = self.path_index[best_solution_neighbour_index - 1]  # For self.adjacent
            self.adjacent[node_id] = [other_neighbour, best_solution_neighbour]
            self.adjacent[best_solution_neighbour] = [node_id, self.adjacent[best_solution_neighbour][1]]
            self.adjacent[other_neighbour] = [self.adjacent[other_neighbour][0], node_id]

            self.path_index.insert(best_solution_neighbour_index, node_id)
        else:
            other_neighbour = self.path_index[best_solution_neighbour_index + 1]  # For self.adjacent
            self.adjacent[node_id] = [best_solution_neighbour, other_neighbour]
            self.adjacent[best_solution_neighbour] = [self.adjacent[best_solution_neighbour][0], node_id]
            self.adjacent[other_neighbour] = [node_id, self.adjacent[other_neighbour][
                1]]  # [self.adjacent[node_id, other_neighbour][1]]

            self.path_index.insert(best_solution_neighbour_index + 1, node_id)

        self.distance += new_distance - previous_distance
        self.neighbours = self.neighbours[:, self.neighbours[0] != node_id]

        mask_2 = ~np.isin(element=self.nodes.nodes_list[node_id].ordered_nodes_by_distance,
                          test_elements=self.path_index)
        neighbours_nodes = self.nodes.nodes_list[node_id].ordered_nodes_by_distance[mask_2][0:self.neighbour_count]
        # neighbours_nodes = self.nodes.nodes_list[node_id].ordered_nodes_by_distance[1:self.neighbour_count + 1]
        neighbours_distance = self.nodes.nodes_list[node_id].distance_to_nodes[neighbours_nodes]

        new_neighbours = np.vstack((neighbours_nodes, neighbours_distance, np.ones(len(neighbours_nodes)) * node_id))

        # print('Neighbours:\n', self.neighbours)
        # print('New neighbours:\n', new_neighbours)

        self.neighbours = np.hstack((self.neighbours, new_neighbours))
        self.neighbours = self.neighbours[:, np.argsort(self.neighbours[1])]
        self.not_visited_index.remove(node_id)
        # print('path index:',self.path_index)
        # print(f'Not visited index: {self.not_visited_index}')

    def add_node_to_path_2(self, number_nodes_check_when_adding):
        node_id = int(self.neighbours[0, 0])  # El nodo mas cercano de self.neighbours[0, 0]
        mask = self.neighbours[0] == node_id  # Mascara filtrando neighbours
        solution_neighbours = self.neighbours[2, mask].astype(int)[:number_nodes_check_when_adding]  # Neighbours al nodo que quiero ingresar
        distance_diff = np.inf

        for i in solution_neighbours:
            index = self.path_index.index(i)  # index of node i in the solution
            distance_right = self.nodes.nodes_list[self.path_index[index]].distance(
                self.nodes.nodes_list[self.path_index[index + 1]])
            new_distance_right = self.nodes.nodes_list[node_id].distance(
                self.nodes.nodes_list[self.path_index[index]]) + \
                                 self.nodes.nodes_list[node_id].distance(
                                     self.nodes.nodes_list[self.path_index[index + 1]])

            if index != 0:
                distance_left = self.nodes.nodes_list[self.path_index[index]].distance(
                    self.nodes.nodes_list[self.path_index[index - 1]])
                new_distance_left = self.nodes.nodes_list[node_id].distance(
                    self.nodes.nodes_list[self.path_index[index]]) + \
                                    self.nodes.nodes_list[node_id].distance(
                                        self.nodes.nodes_list[self.path_index[index - 1]])
            else:
                distance_left = np.inf
                new_distance_left = np.inf

            if distance_diff > (new_distance_right - distance_right):
                distance_diff = new_distance_right - distance_right
                previous_distance = distance_right
                new_distance = new_distance_right
                left = False
                best_solution_neighbour = i
                best_solution_neighbour_index = index
            if distance_diff > (new_distance_left - distance_left):
                distance_diff = new_distance_left - distance_left
                previous_distance = distance_left
                new_distance = new_distance_left
                left = True
                best_solution_neighbour = i
                best_solution_neighbour_index = index

        if left:
            other_neighbour = self.path_index[best_solution_neighbour_index - 1]  # For self.adjacent
            self.adjacent[node_id] = [other_neighbour, best_solution_neighbour]
            self.adjacent[best_solution_neighbour] = [node_id, self.adjacent[best_solution_neighbour][1]]
            self.adjacent[other_neighbour] = [self.adjacent[other_neighbour][0], node_id]

            self.path_index.insert(best_solution_neighbour_index, node_id)
        else:
            other_neighbour = self.path_index[best_solution_neighbour_index + 1]  # For self.adjacent
            self.adjacent[node_id] = [best_solution_neighbour, other_neighbour]
            self.adjacent[best_solution_neighbour] = [self.adjacent[best_solution_neighbour][0], node_id]
            self.adjacent[other_neighbour] = [node_id, self.adjacent[other_neighbour][1]]  # [self.adjacent[node_id, other_neighbour][1]]

            self.path_index.insert(best_solution_neighbour_index + 1, node_id)

        self.distance += new_distance - previous_distance
        self.neighbours = self.neighbours[:, self.neighbours[0] != node_id]

        mask_2 = ~np.isin(element=self.nodes.nodes_list[node_id].ordered_nodes_by_distance,
                          test_elements=self.path_index)  # Mask of nodes not in solution for ordered by distance list
        # print(f'path index: {self.path_index}')
        # print(f'new node: {node_id}')
        # print(f'ordered nodes by distance: {self.nodes.nodes_list[node_id].ordered_nodes_by_distance}')
        # print('mask_2: ', mask_2)
        neighbours_nodes = self.nodes.nodes_list[node_id].ordered_nodes_by_distance[mask_2][0:self.neighbour_count]
        # print(self.nodes.nodes_list[node_id].ordered_nodes_by_distance[mask_2])
        # print(neighbours_nodes)
        # _ = input('...')
        neighbours_distance = self.nodes.nodes_list[node_id].distance_to_nodes[neighbours_nodes]

        new_neighbours = np.vstack((neighbours_nodes, neighbours_distance, np.ones(len(neighbours_nodes)) * node_id))

        self.neighbours = np.hstack((self.neighbours, new_neighbours))
        self.neighbours = self.neighbours[:, np.argsort(self.neighbours[1])]
        self.not_visited_index.remove(node_id)

    def adjacent_verification(self):
        for i in range(1, len(self.path_index) - 1):
            condition = (self.path_index[i - 1] == self.adjacent[self.path_index[i]][0]) & (
                        self.path_index[i + 1] == self.adjacent[self.path_index[i]][1])
            if not condition:
                print('error de adjacente en nodo:', self.path_index[i])
                print(f'path index: {self.path_index}')
                print(f'adjacent {[[i, self.adjacent[i]] for i in self.path_index]}')
                _ = input('')
        print('pass verification')

    def two_opt(self):
        for node_id in self.path_index:
            node = self.nodes.nodes_list[node_id]
            right_node_id = self.adjacent[node_id][1]
            left_node_id = self.adjacent[node_id][0]
            # Define sections of nodes neighbouring the node being analyzed
            unfiltered_neighbour_section_list_left = [[self.adjacent[j][0], j] for j in
                                                      node.ordered_nodes_by_distance[1:self.neighbour_count] if
                                                      self.adjacent[j][0] != node_id]
            unfiltered_neighbour_section_list_right = [[j, self.adjacent[j][1]] for j in
                                                       node.ordered_nodes_by_distance[1:self.neighbour_count] if
                                                       self.adjacent[j][1] != node_id]
            unfiltered_neighbour_section = np.array(unfiltered_neighbour_section_list_right +
                                                    unfiltered_neighbour_section_list_left)
            neighbour_section = np.unique(np.sort(unfiltered_neighbour_section, axis=1), axis=0)

            for section in neighbour_section:

                # Will evaluate two sections, the one from the node being analyzed and its right node, and another
                # section, that's why we check that right_node_id is not in the section analyzed
                # There is room from improvement between this if statement and the previous lists made.

                if np.isin(section, right_node_id).any():
                    continue

                # node
                right_node = self.nodes.nodes_list[right_node_id]
                neighbour_node_c = self.nodes.nodes_list[section[1]]
                neighbour_node_d = self.nodes.nodes_list[section[0]]

                p1 = nodos.Point(node.x_cord, node.y_cord)
                p2 = nodos.Point(right_node.x_cord, right_node.y_cord)
                p3 = nodos.Point(neighbour_node_c.x_cord, neighbour_node_c.y_cord)
                p4 = nodos.Point(neighbour_node_d.x_cord, neighbour_node_d.y_cord)

                condition_1 = nodos.intersect(p1, p2, p3, p4)  # Checks if the sections p1p2 and p3p4 do cross

                if condition_1:

                    node_index = self.path_index.index(node_id)
                    right_node_index = self.path_index.index(right_node.node_id)
                    neighbour_node_c_index = self.path_index.index(neighbour_node_c.node_id)
                    neighbour_node_d_index = self.path_index.index(neighbour_node_d.node_id)

                    loop_start_index = min(node_index, right_node_index, neighbour_node_c_index, neighbour_node_d_index)
                    loop_start_inner_index = loop_start_index + 1
                    loop_end_index = max(node_index, right_node_index, neighbour_node_c_index, neighbour_node_d_index)
                    loop_end_inner_index = loop_end_index - 1

                    loop_start_id = self.path_index[loop_start_index]
                    loop_start_inner_id = self.path_index[loop_start_inner_index]
                    loop_end_id = self.path_index[loop_end_index]
                    loop_end_inner_id = self.path_index[loop_end_inner_index]

                    loop_start_node = self.nodes.nodes_list[loop_start_id]
                    loop_start_inner_node = self.nodes.nodes_list[loop_start_inner_id]
                    loop_end_node = self.nodes.nodes_list[loop_end_id]
                    loop_end_inner_node =  self.nodes.nodes_list[loop_end_inner_id]


                    # # Prepare figure
                    # plt.figure(figsize=(10, 10))
                    # plt.title(f'{condition_1}')
                    # # plt.xlim((0, 70))
                    # # plt.ylim((0, 70))
                    # # Plot all points and sections
                    # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #          [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # plt.scatter([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #             [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # # Plot in red the neighbour sections
                    # for i in neighbour_section:
                    #     plt.plot([self.nodes.nodes_list[j].x_cord for j in i],
                    #              [self.nodes.nodes_list[j].y_cord for j in i],
                    #              c='y')
                    # # Plot the loop
                    # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index[loop_start_index:loop_end_index]],
                    #          [self.nodes.nodes_list[i].y_cord for i in self.path_index[loop_start_index:loop_end_index]], c='r')
                    # # Plot in yellow the neighbour nodess
                    # for j in node.ordered_nodes_by_distance[1:self.neighbour_count]:
                    #     plt.scatter(self.nodes.nodes_list[j].x_cord, self.nodes.nodes_list[j].y_cord, c='y')
                    # # Print the node_k in black
                    # plt.scatter(self.nodes.nodes_list[node_id].x_cord, self.nodes.nodes_list[node_id].y_cord,
                    #             c='black')
                    #
                    # plt.plot([node.x_cord, right_node.x_cord], [node.y_cord, right_node.y_cord], c='black')
                    # plt.plot([neighbour_node_c.x_cord, neighbour_node_d.x_cord], [neighbour_node_c.y_cord, neighbour_node_d.y_cord], c='black')
                    # plt.show()
                    # plt.close()
                    #
                    # print('cross found:')
                    #
                    # # plt.figure(figsize=(10, 10))
                    # # # plt.xlim((0, 70))
                    # # # plt.ylim((0, 70))
                    # # # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index[loop_start_index:loop_end_index+1]],
                    # # #          [self.nodes.nodes_list[i].y_cord for i in self.path_index[loop_start_index:loop_end_index+1]])
                    # # # plt.scatter([self.nodes.nodes_list[i].x_cord for i in self.path_index[loop_start_index:loop_end_index+1]],
                    # # #             [self.nodes.nodes_list[i].y_cord for i in self.path_index[loop_start_index:loop_end_index+1]])
                    # # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    # #          [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # # plt.scatter([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    # #             [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # #
                    # # plt.savefig(f'images\loop_{n}.png', dpi=200)

                    #loop_start_inner_id = self.adjacent[self.path_index[loop_start_index]][1]
                    # loop_end_inner_id = self.adjacent[self.path_index[loop_end_index]][0]

                    # print(self.path_index)
                    # print('---------------------------')
                    # print([f'{i} : {self.adjacent[i]}' for i in self.path_index])
                    # _ = input('')
                    # self.adjacent_verification()

                    # print(f'changing adjacent {self.path_index[loop_start_index]} for {[self.adjacent[self.path_index[loop_start_index]][0], loop_end_inner_id]}')

                    self.adjacent[loop_start_id] = [self.adjacent[loop_start_id][0], loop_end_inner_id]
                    self.adjacent[loop_end_id] = [loop_start_inner_id, self.adjacent[loop_end_id][1]]
                    self.adjacent[loop_start_inner_id] = [self.adjacent[loop_start_inner_id][1], loop_end_id]
                    self.adjacent[loop_end_inner_id] = [loop_start_id, self.adjacent[loop_end_inner_id][0]]

                    for swapped_node in self.path_index[loop_start_inner_index + 1:loop_end_inner_index]:
                        self.adjacent[swapped_node] = self.adjacent[swapped_node][::-1]

                    # print('loop start', loop_start_index)
                    self.path_index = self.path_index[:loop_start_inner_index] +\
                                      self.path_index[loop_end_inner_index:loop_start_index:-1] +\
                                      self.path_index[loop_end_index:]
                    self.distance += -loop_start_node.distance_to_nodes[loop_start_inner_id]\
                                     + loop_start_node.distance_to_nodes[loop_end_inner_id]\
                                     - loop_end_node.distance_to_nodes[loop_end_inner_id]\
                                     + loop_end_node.distance_to_nodes[loop_start_inner_id]

                    # print(self.path_index)
                    # print('---------------------------')
                    # print([f'{i} : {self.adjacent[i]}' for i in self.path_index])
                    # _ = input('')
                    # self.adjacent_verification()
                    #
                    # # Prepare figure
                    # plt.figure(figsize=(10, 10))
                    # plt.title(f'node: {node_id}')
                    # # plt.xlim((0, 70))
                    # # plt.ylim((0, 70))
                    # # Plot all points and sections
                    # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #          [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # plt.scatter([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #             [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # # # Plot in red the neighbour sections
                    # # for i in neighbour_section:
                    # #     plt.plot([self.nodes.nodes_list[j].x_cord for j in i],
                    # #              [self.nodes.nodes_list[j].y_cord for j in i],
                    # #              c='y')
                    # # # Plot the loop
                    # # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index[loop_start_index:loop_end_index]],
                    # #          [self.nodes.nodes_list[i].y_cord for i in self.path_index[loop_start_index:loop_end_index]], c='r')
                    # # Plot in yellow the neighbour nodess
                    # # for j in node_k.ordered_nodes_by_distance[1:self.neighbour_count]:
                    # #     plt.scatter(self.nodes.nodes_list[j].x_cord, self.nodes.nodes_list[j].y_cord, c='y')
                    # # Print the node_k in black
                    # # plt.scatter(self.nodes.nodes_list[k].x_cord, self.nodes.nodes_list[k].y_cord, c='black')
                    #
                    # # plt.plot([a.x_cord, b.x_cord], [a.y_cord, b.y_cord], c='black')
                    # # plt.plot([c.x_cord, d.x_cord], [c.y_cord, d.y_cord], c='black')
                    # plt.show()
                    # plt.close()

                    # self.distance +=
                    break

    def swap_node_to_between(self):
        for node_id in self.path_index[1:-1]:#range(1,len(self.path_index)-1):
            #print(f'path index: {self.path_index}')
            #node_id = self.path_index[node_index]
            node = self.nodes.nodes_list[node_id]
            node_index = self.path_index.index(node_id)

            original_left_node_id = self.adjacent[node_id][0]
            original_right_node_id = self.adjacent[node_id][1]
            distance_original_node = node.distance_to_nodes[original_left_node_id] +\
                                     node.distance_to_nodes[original_right_node_id]
            distance_new_where_node = self.nodes.nodes_list[original_left_node_id].distance_to_nodes[original_right_node_id]

            neighbours_list_id = node.ordered_nodes_by_distance[1:self.neighbour_count + 1]
            neighbours_list_left_id = [self.adjacent[neighbour_id][0] for neighbour_id in neighbours_list_id]
            neighbours_list_right_id = [self.adjacent[neighbour_id][1] for neighbour_id in neighbours_list_id]

            delta_length_left = []
            delta_length_right = []
            for i in range(self.neighbour_count):  # Instead of generating two comprehension list it's done in a for.
                neighbour_id = neighbours_list_id[i]
                left_neighbour_id = neighbours_list_left_id[i]
                right_neighbour_id = neighbours_list_right_id[i]

                distance_original_neighbour_left = self.nodes.nodes_list[neighbour_id].distance_to_nodes[left_neighbour_id]
                distance_original_neighbour_right = self.nodes.nodes_list[neighbour_id].distance_to_nodes[right_neighbour_id]

                distance_new_neighbour_left = self.nodes.nodes_list[neighbour_id].distance_to_nodes[node_id] +\
                                                   self.nodes.nodes_list[node_id].distance_to_nodes[left_neighbour_id]\
                    if node_id != left_neighbour_id else float('inf')
                distance_new_neighbour_right = self.nodes.nodes_list[neighbour_id].distance_to_nodes[node_id] +\
                                                   self.nodes.nodes_list[node_id].distance_to_nodes[right_neighbour_id]\
                    if node_id != right_neighbour_id else float('inf')

                delta_length_left.append(distance_new_neighbour_left
                                         - distance_original_neighbour_left
                                         + distance_new_where_node
                                         - distance_original_node)
                delta_length_right.append(distance_new_neighbour_right
                                          - distance_original_neighbour_right
                                          + distance_new_where_node
                                          - distance_original_node)

            delta_length_left = np.array(delta_length_left)
            delta_length_right = np.array(delta_length_right)

            # print(np.array([neighbours_list_id, neighbours_list_left_id, neighbours_list_right_id, delta_length_left, delta_length_right]).T)

            left = np.min(delta_length_left) < np.min(delta_length_right)

            if left:
                index_min = np.argmin(delta_length_left)
                neighbour_id = neighbours_list_id[index_min]
                neighbour_left_id = neighbours_list_left_id[index_min]
                neighbour_index_path = self.path_index.index(neighbour_id)
                delta_length = delta_length_left[index_min]
                if delta_length < 0:
                    # print(f'path index: {self.path_index}')
                    # print(f'adjacent {[[i, self.adjacent[i]] for i in self.path_index]}')
                    self.distance += delta_length
                    # ver que es mas chico si node_index o la nueva posicion....
                    if node_index > neighbour_index_path:
                        self.path_index.insert(neighbour_index_path, self.path_index.pop(node_index))
                    else:
                        self.path_index.insert(neighbour_index_path-1, self.path_index.pop(node_index))
                    # if node_index > neighbour_index_path:
                    #     del self.path_index[node_index]
                    #     self.path_index.insert(neighbour_index_path, node_id)
                    # else:
                    #     self.path_index.insert(neighbour_index_path, node_id)
                    #     del self.path_index[node_index]
                    self.adjacent[original_left_node_id][1] = original_right_node_id
                    self.adjacent[original_right_node_id][0] = original_left_node_id

                    self.adjacent[node_id] = [neighbour_left_id, neighbour_id]

                    self.adjacent[neighbour_left_id][1] = node_id
                    self.adjacent[neighbour_id][0] = node_id


            else:
                index_min = np.argmin(delta_length_right)
                neighbour_id = neighbours_list_id[index_min]
                neighbour_right_id = neighbours_list_right_id[index_min]
                neighbour_index_path = self.path_index.index(neighbour_id)
                delta_length = delta_length_right[index_min]
                # print(f'delta lenght = {delta_length}')
                if delta_length < 0:
                    # print(f'path index: {self.path_index}')
                    # print(f'adjacent {[[i, self.adjacent[i]] for i in self.path_index]}')
                    self.distance += delta_length
                    # ver que es mas chico si node_index o la nueva posicion....
                    if node_index > neighbour_index_path:
                        self.path_index.insert(neighbour_index_path+1, self.path_index.pop(node_index))
                    else:
                        self.path_index.insert(neighbour_index_path, self.path_index.pop(node_index))                    # if node_index > neighbour_index_path:

                    self.adjacent[original_left_node_id][1] = original_right_node_id
                    self.adjacent[original_right_node_id][0] = original_left_node_id

                    self.adjacent[node_id] = [neighbour_id, neighbour_right_id]

                    self.adjacent[neighbour_right_id][0] = node_id
                    self.adjacent[neighbour_id][1] = node_id


    def k_opt(self, t2_node_id, t1_node_id, k, x_y_array):
        if k > 0:  # k is the number of ops to be checked, method is recursive.

            # print(self.path_index, len(self.path_index))

            self.copy()  # Apply the method and make a copy of my solution. Stored at the solution list.
            k_solution = Solution.solution_list[-1]  # store my copied solution in a variable, it exists on the list.
            t2_adjacent = self.adjacent[t2_node_id]  # get the adjacent nodes to t2 in a variable.
            t2_neighbours = self.nodes.nodes_list[t2_node_id].ordered_nodes_by_distance[1:Solution.recent_k_opt_nodes_number + 4]# list of t2 neighbours, using the same amount of nodes as stored in recently visited + 2 in case of adjacent.
            new_t2_neighbour_id = [x for x in t2_neighbours if ((x not in t2_adjacent) & (x not in Solution.recent_k_opt_nodes))][0]  # Get the first neighbour that is not adjacent nor visited.
            next_t1_neighbour_id = self.adjacent[new_t2_neighbour_id][0 if (self.adjacent[t1_node_id][1] == t2_node_id) else 1]

            # The new t1 neighbour was conected to t2,
            # so if t1 is to the right of t2, then the one
            # to the left of t2 is the new t1 neighbour.

            t1_node_index = self.path_index.index(t1_node_id)
            t2_node_index = self.path_index.index(t2_node_id)
            new_t2_neighbour_index = self.path_index.index(new_t2_neighbour_id)

            if self.adjacent[t1_node_id][1] == t2_node_id:  # Check if t2 was to the right of t1
                if new_t2_neighbour_index != 0:
                    next_t1_neighbour_index = new_t2_neighbour_index - 1
                else:
                    next_t1_neighbour_index =  self.node_count - 2  # self.path_index.index(self.adjacent[0][0])  # self.adjacent[0][0]
            else:
                next_t1_neighbour_index = new_t2_neighbour_index + 1

            # Tengo que rearmar el path.
            # En vez de buscar el primero y ultimo podria buscar los dos del medio, mas facil y me evito verificar lo del 0
            index_order_list = [t1_node_index, t2_node_index, new_t2_neighbour_index, next_t1_neighbour_index]
            index_order_list.sort()

            # This is for a tricky scenario. It es for the case node 0 and the last node are among the nodes, the script
            # needs to refer two the second appearance of node 0 but as index returns the first instance we need this if

            if (0 in index_order_list) & (self.node_count - 1 in index_order_list) :
                index_order_list.remove(0)
                index_order_list.append(self.node_count)
            #     flag = True
            #     print(index_order_list)
            #     print(f't1: {t1_node_id}, t2: {t2_node_id}, new_t1_neighbour: {new_t2_neighbour_id}, new_t2_neighbour: {next_t1_neighbour_id}')
            # else:
            #     flag = False

            # index_order_list.remove(0) if (0 in index_order_list) else None# Cheto el if en una linea
            loop_start_inner_index = index_order_list[1]
            loop_end_inner_index = index_order_list[2]
            loop_start_index = loop_start_inner_index - 1
            loop_end_index = loop_end_inner_index + 1

            loop_start_id = self.path_index[loop_start_index]
            loop_start_inner_id = self.path_index[loop_start_inner_index]
            loop_end_id = self.path_index[loop_end_index] # un error aca!!
            loop_end_inner_id = self.path_index[loop_end_inner_index]

            loop_start_node = self.nodes.nodes_list[loop_start_id]
            loop_start_inner_node = self.nodes.nodes_list[loop_start_inner_id]
            loop_end_node = self.nodes.nodes_list[loop_end_id]
            loop_end_inner_node = self.nodes.nodes_list[loop_end_inner_id]

            k_solution.adjacent[loop_start_id] = [k_solution.adjacent[loop_start_id][0], loop_end_inner_id]
            k_solution.adjacent[loop_end_id] = [loop_start_inner_id, k_solution.adjacent[loop_end_id][1]]
            k_solution.adjacent[loop_start_inner_id] = [k_solution.adjacent[loop_start_inner_id][1], loop_end_id]
            k_solution.adjacent[loop_end_inner_id] = [loop_start_id, k_solution.adjacent[loop_end_inner_id][0]]

            for swapped_node in k_solution.path_index[loop_start_inner_index + 1:loop_end_inner_index]:  # da vuelta los adjacent del rulo
                k_solution.adjacent[swapped_node] = k_solution.adjacent[swapped_node][::-1]

            # print('loop start', loop_start_index)

            # doy vuelta la solucion
            k_solution.path_index = k_solution.path_index[:loop_start_inner_index] + \
                                    k_solution.path_index[loop_end_inner_index:loop_start_index:-1] + \
                                    k_solution.path_index[loop_end_index:]

            # re calculo la solucion
            k_solution.distance += -loop_start_node.distance_to_nodes[loop_start_inner_id] \
                                   + loop_start_node.distance_to_nodes[loop_end_inner_id] \
                                   - loop_end_node.distance_to_nodes[loop_end_inner_id] \
                                   + loop_end_node.distance_to_nodes[loop_start_inner_id]
            Solution.recent_k_opt_nodes.append(new_t2_neighbour_id)

            # if flag:
            #     print(self.path_index)
            #     print(k_solution.path_index)

            if len(Solution.recent_k_opt_nodes) > Solution.recent_k_opt_nodes_number:
                del Solution.recent_k_opt_nodes[0]
            # print(Solution.recent_k_opt_nodes)
            # print(f'path_index: {k_solution.path_index}')
            # k_solution.plot_solution(x_y_array, title=f'k_opt, distance={round(k_solution.distance,2)}, k={k}')
            k_solution.k_opt(next_t1_neighbour_id, t1_node_id, k-1, x_y_array)

        else:
            Solution.calc_solution_distances()
            #print('distances: ', Solution.solution_distances)
            Solution.solution_list = [Solution.solution_list[Solution.solution_distances.index(min(Solution.solution_distances))]]
