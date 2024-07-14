from typing import List
import matplotlib.pyplot as plt
#from solver import Customer
from collections import namedtuple
#Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

import matplotlib.colors as mcolors

from classes import Vehicle, Customer

def generate_colors(n):
    cmap = plt.cm.get_cmap('hsv', n)
    norm = mcolors.Normalize(vmin=0, vmax=n-1)
    colors = [cmap(norm(i)) for i in range(n)]
    return colors

# # Example usage:
# n_colors = 5
# colors = generate_colors(n_colors)

# print(colors)

def plot_solution(customers:list[Customer], vehicle_tours: list[list[Customer]] = None, depot: Customer = None):
    # add colours for each route.
    
    vehicle_tours = [vehicle_tour for vehicle_tour in vehicle_tours if len(vehicle_tour)>0]
    
    colours = generate_colors(len(vehicle_tours)+1)
    
    fig, ax = plt.subplots()

    x_values = [customer.x for customer in customers]
    y_values = [customer.y for customer in customers]

    ax.scatter(x_values, y_values)
    
    if vehicle_tours:
        for i, vehicle_tour in enumerate(vehicle_tour for vehicle_tour in vehicle_tours if len(vehicle_tour) > 0):
            #print(colours[i])
            x_values = [depot.x] + [customer.x for customer in vehicle_tour] + [depot.x]
            y_values = [depot.y] + [customer.y for customer in vehicle_tour] + [depot.y]
            ax.plot(x_values, y_values, color=colours[i])

    plt.show()
    
    return fig

def plot_solution_from_objects(customers:List['Customer'], vehicles: List['Vehicle'] = None, depot: Customer = None):
    # add colours for each route.
    
    used_vehicles = [vehicle for vehicle in vehicles if len(vehicle.route)>0]
    
    colours = generate_colors(len(used_vehicles)+1)
    
    fig, ax = plt.subplots()

    x_values = [customer.x for customer in customers]
    y_values = [customer.y for customer in customers]

    ax.scatter(x_values, y_values)
    
    for customer in customers:
        ax.annotate(customer.index, (customer.x, customer.y), textcoords="offset points", xytext=(5,5), ha='right')
    
    if used_vehicles:
        for i, vehicle in enumerate(vehicles): #for vehicle_tour in vehicle_tours if len(vehicle_tour) > 0):
            x_values = [depot.x] + [customer.x for customer in vehicle.route] + [depot.x]
            y_values = [depot.y] + [customer.y for customer in vehicle.route] + [depot.y]
            ax.plot(x_values, y_values, color=colours[i])

    plt.show()
    #plt.savefig('../images/fig.png', dpi=20)
    
    
    return fig