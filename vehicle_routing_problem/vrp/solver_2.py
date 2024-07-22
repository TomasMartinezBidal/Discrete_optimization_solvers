#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import classes

from plotter import plot_solution, plot_solution_from_objects
import numpy as np

from pulp import *


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    print(f'customer_count: {customer_count}', f'vehicle_count: {vehicle_count}', f'vehicle_capacity: {vehicle_capacity}', sep="\n")
    
    #Customers = classes.Customer
    
    # create depot
    line = lines[1]
    parts = line.split()
    depot = classes.Customer(x=float(parts[1]), y=float(parts[2]), depot=True)
    
    customer_collection = classes.Customer_collection([depot] + [ classes.Customer(i-1, int(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2])) for i in range(2, customer_count+1)])
    
    # get usefull matrix of customers
    customer_collection.calc_distances()
    #Customers.generate_customer_list_by_demand()

    first_fleet = classes.Vehicle_fleet([classes.Vehicle(i, vehicle_capacity, depot) for i in range(vehicle_count)])

    
    solution = classes.Solution(customer_count=customer_count,
                                distance_array=[],
                                distance_array_with_depot=customer_collection.distances)
    
    # Asing customers to vehicles, it assings customers to the first vehicle with available capacity
    # the vehicle cant be choosen randomly because in some cases customers may not fit.
    # customers are also sorted by capacity to ensure the all fit.
    for customer in customer_collection.customer_list_w_depot[1:]:
        available_vehicles = [vehicle for vehicle in first_fleet.vehicle_list if vehicle.remaining_capacity >= customer.demand]
        solution.add_customer_to_vehicle(available_vehicles[0], customer)
    
    # print('remaining capacities',[vehicle.remaining_capacity for vehicle in first_fleet.vehicle_list])

    print('initial cost:',solution.calc_cost())
    
    # prepare the solution in the specified output format

    # print(solution.routes)
    plot_solution_from_objects(customers=customer_collection.customer_list_w_depot, vehicles=first_fleet.vehicle_list, depot=depot)
    
    # make pulp solver.
    
    #Pending, select a set of vehicles
    
    vehicle_subset: List[classes.Vehicle] = np.random.choice(first_fleet.vehicle_list,size=2, replace=False)
    customers_subset = [customer for vehicle in vehicle_subset for customer in vehicle.route]
    
    
    
    prob = LpProblem("CVRP", const.LpMinimize)
    
    customer_indexes = list(range(1, customer_count))
    customer_w_warehouse_indexes = list(range(customer_count))
    customer_index_no_diag = [(i,j) for i in customer_w_warehouse_indexes for j in customer_w_warehouse_indexes if i != j]
    # decition variables
    pulp_routes_object = pulp.LpVariable.dicts("Used_routes", customer_index_no_diag, cat="Binary")
    solution.u_vars = pulp.LpVariable.dicts("u vars", customer_indexes, cat="Intiger")
    # for i in range(customer_count):
    #     for j in range(customer_count):
    #         if i != j:
    #             solution.routes[i,j] = pulp.LpVariable(name=f'route({i,j})', cat='Binary')
    
    # for i in customer_indexes:
    #     solution.u_vars[i] = pulp.LpVariable(name=f'u var {i}', cat='Intiger')

    # objective function
    prob += pulp.lpSum([solution.distance_array_with_depot[i,j] * pulp_routes_object[i,j] for i, j in customer_index_no_diag])
    
    # restriction  
    prob += pulp.lpSum([pulp_routes_object[0,j] for j in customer_indexes]) <= vehicle_count
    prob += pulp.lpSum([pulp_routes_object[i,0] for i in customer_indexes]) <= vehicle_count
    for i in customer_indexes:
        prob += pulp.lpSum([pulp_routes_object[i,j] for j in customer_w_warehouse_indexes if j!= i]) == 1
    for j in customer_indexes:
        prob += pulp.lpSum([pulp_routes_object[i,j] for i in customer_w_warehouse_indexes if j!= i]) == 1
    
    M = 1000  # A sufficiently large number
    for i,j in customer_index_no_diag:
        if i != 0 and j != 0:
            prob += solution.u_vars[i] + customer_collection.customer_list_w_depot[j].demand - solution.u_vars[j] <= M * (1 - pulp_routes_object[i, j])
            prob += solution.u_vars[i] + customer_collection.customer_list_w_depot[j].demand - solution.u_vars[j] >= - M * (1 -pulp_routes_object[i, j])
    
    for i in customer_indexes:
        prob += solution.u_vars[i] >= customer_collection.customer_list_w_depot[i].demand
        prob += solution.u_vars[i] <= vehicle_capacity
        
    prob.solve(pulp.PULP_CBC_CMD(msg=1, warmStart=True, timeLimit=180))
    
    print(f'cost pulp: {pulp.value(prob.objective)}')
    for i,j in customer_index_no_diag:
        solution.routes[i,j] = round(pulp.value(pulp_routes_object[i,j]))
    
    number_of_used_vehicles = np.sum(solution.routes[0], dtype=int)
    customer_visited_from_depot = np.where(solution.routes[0])[0].tolist()
    
    #retore vehicles
    for vehicle in first_fleet.vehicle_list:
        vehicle.route = []
    
    for i in range(number_of_used_vehicles):
        vehicle = first_fleet.vehicle_list[i]
        first_customer_index = customer_visited_from_depot[i]
        vehicle.route.append(customer_collection.customer_list_w_depot[0])
        vehicle.route.append(customer_collection.customer_list_w_depot[first_customer_index])
        last_added_customer_index = first_customer_index
        while True:
            next_customer_index = np.where(solution.routes[last_added_customer_index])[0].tolist()[0]
            vehicle.route.append(customer_collection.customer_list_w_depot[next_customer_index])
            last_added_customer_index = next_customer_index
            if last_added_customer_index == 0:
                break
      
    plot_solution_from_objects(customers=customer_collection.customer_list_w_depot, vehicles=first_fleet.vehicle_list, depot=depot)
            
        
    
    outputData = '%.2f' % solution.calc_cost() + ' ' + str(0) + '\n'
    for vehicle in first_fleet.vehicle_list:
        outputData += ' '.join([str(customer.index) for customer in vehicle.route]) + ' ' + '\n'
    # print(f'output data: {outputData}')
    return outputData

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

