import time
import nodos
import numpy as np
import math
import matplotlib.pyplot as plt


class Solution:

    def __init__(self, node_count, nodes, neighbour_count):
        self.path_index = []
        self.neighbours = []
        self.not_visited_index = list(range(1, node_count))
        self.distance = 0
        self.neighbour_count = neighbour_count
        self.nodes = nodes
        self.adjacent = {}

    def plot_solution(self, coord_array, title='no_title'):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.plot(coord_array[self.path_index, 0], coord_array[self.path_index, 1])
        plt.scatter(coord_array[:, 0], coord_array[:, 1])
        for x, y, text in zip([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                              [self.nodes.nodes_list[i].y_cord for i in self.path_index],
                              self.path_index[:-1]):
            plt.annotate(text, (x, y + 0.1))
        plt.savefig(f'images\\{title}.png', dpi=200)
        plt.close()

    def add_zero_to_path(self):
        if self.path_index == []:
            self.path_index = [0, 0]
            neighbours_nodes = self.nodes.nodes_list[0].ordered_nodes_by_distance[1:self.neighbour_count + 1]
            neighbours_distance = self.nodes.nodes_list[0].distance_to_nodes[neighbours_nodes]
            neighbours_of = np.zeros(len(neighbours_nodes))
            neighbours = np.vstack((neighbours_nodes, neighbours_distance, neighbours_of))
            self.neighbours = neighbours
            self.adjacent[0] = [0, 0]

    def add_node_to_path(self):
        node_id = int(self.neighbours[0, 0])
        mask = self.neighbours[0] == node_id
        solution_neighbours = self.neighbours[2, mask].astype(int)
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

    def adjacent_verification(self):
        for i in range(1, len(self.path_index) - 1):
            condition = (self.path_index[i - 1] == self.adjacent[self.path_index[i]][0]) & (
                        self.path_index[i + 1] == self.adjacent[self.path_index[i]][1])
            if not condition:
                print('error de adjacente en nodo:', self.path_index[i])
                print(f'path index: {self.path_index}')
                print(f'adjacent {[[i, self.adjacent[i]] for i in self.path_index]}')
                _ = input('')

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

                    # print(f'optimized via left {[node_id,neighbour_id,original_left_node_id,original_right_node_id,neighbour_left_id]}')
                    # print(f'path index: {self.path_index}')

                    # plt.figure(figsize=(10, 10))
                    # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index], [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # plt.scatter([self.nodes.nodes_list[i].x_cord for i in self.path_index], [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # for x, y, text in zip([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #                       [self.nodes.nodes_list[i].y_cord for i in self.path_index],
                    #                       self.path_index[:-1]):
                    #     plt.annotate(text, (x, y+0.1))
                    # plt.show()

                    # self.adjacent_verification()


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
                    #     del self.path_index[node_index]
                    #     # print(f'path index: {self.path_index}')
                    #     self.path_index.insert(neighbour_index_path+1, node_id)
                    # else:
                    #     self.path_index.insert(neighbour_index_path, node_id)
                    #     # print(f'path index: {self.path_index}')
                    #     del self.path_index[node_index]
                    self.adjacent[original_left_node_id][1] = original_right_node_id
                    self.adjacent[original_right_node_id][0] = original_left_node_id

                    self.adjacent[node_id] = [neighbour_id, neighbour_right_id]

                    self.adjacent[neighbour_right_id][0] = node_id
                    self.adjacent[neighbour_id][1] = node_id
                    # print(f'optimized via right {[node_id,neighbour_id,original_left_node_id,original_right_node_id,neighbour_right_id]}')
                    # print(f'path index: {self.path_index}')

                    # plt.figure(figsize=(10, 10))
                    # plt.plot([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #          [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # plt.scatter([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #             [self.nodes.nodes_list[i].y_cord for i in self.path_index])
                    # for x, y, text in zip([self.nodes.nodes_list[i].x_cord for i in self.path_index],
                    #                       [self.nodes.nodes_list[i].y_cord for i in self.path_index],
                    #                       self.path_index[:-1]):
                    #     plt.annotate(text, (x, y+0.1))
                    # plt.show()

                    # self.adjacent_verification()
