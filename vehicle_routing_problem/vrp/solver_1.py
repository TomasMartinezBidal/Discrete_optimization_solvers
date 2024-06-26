#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
#from test import test_function
#test_function()
import classes

from plotter import plot_solution, plot_solution_from_objects
import numpy as np

#Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

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
    
    Customers = classes.Customer
    
    # create depot
    line = lines[1]
    parts = line.split()
    depot = Customers(x=float(parts[1]), y=float(parts[2]), depot=True)
    
    #create customers
    for i in range(1, customer_count):
        line = lines[i+1]
        parts = line.split()
        Customers(i, int(parts[0]), float(parts[1]), float(parts[2]))
    
    # get usefull matrix of customers
    Customers.clac_distances()
    Customers.generate_customer_list_by_demand()
    
    # create vehicles
    Vehicles = classes.Vehicle
    for i in range(vehicle_count):
        Vehicles(i, vehicle_capacity, depot)
    
    solution = classes.Solution(customer_count=customer_count,
                                distance_array=Customers.distances,
                                distance_array_with_depot=Customers.distances_with_depot)
    
    # Asing customers to vehicles, it assings customers to the first vehicle with available capacity
    # the vehicle cant be choosen randomly because in some cases customers may not fit.
    # customers are also sorted by capacity to ensure the all fit.
    for customer in Customers.customer_list_by_demand:
        available_vehicles = [vehicle for vehicle in Vehicles.vehicle_list if vehicle.remaining_capacity >= customer.demand]
        solution.add_to_vehicle(available_vehicles[0], customer)
    
    print([vehicle.remaining_capacity for vehicle in Vehicles.vehicle_list])

    # prepare the solution in the specified output format
    outputData = '%.2f' % solution.calc_cost() + ' ' + str(0) + '\n'
    for vehicle in Vehicles.vehicle_list:
        outputData += ' '.join([str(customer.index) for customer in vehicle.route]) + ' ' + '\n'

    plot_solution_from_objects(customers=Customers.customer_list, vehicles=Vehicles.vehicle_list, depot=depot)
    
    return outputData


    #the depot is always the first customer in the input
    depot = customers[0] 

    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:  # at least on customer can fit.
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    plot_solution(customers=customers, vehicle_tours=vehicle_tours, depot=depot)

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

