import numpy as np
import scipy.spatial as sc
from typing import List


class Customer:
    customer_list: List['Customer'] = []
    y_coords = np.array([])
    x_coords = np.array([])
    total_nodes = 0
    
    def __init__(self, index : int = None, demand : int = None, x : float = None, y : float = None, depot: bool= False) -> None:
        self.x = x
        self.y = y
        if not depot:
            self.index = index
            self.demand = demand
            self.ordered_customers_distance = []
            Customer.customer_list.append(self)
            Customer.x_coords = np.append(Customer.x_coords, x)
            Customer.y_coords = np.append(Customer.y_coords, y)
            Customer.total_nodes += 1
    
    @classmethod
    def clac_distances(cls): 
        array_x_y = np.array((cls.x_coords, cls.y_coords)).T
        distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
        cls.ordered_customers_distance = np.argsort(distances)
        for i, customer in enumerate(cls.customer_list):
            customer.distance_to_customers = distances[i]
            customer.ordered_customers_distance = distances[i]
            
        
class Vehicle:
    cost = 0
    vehicle_list: List['Vehicle'] = []
    
    def __init__(self, index: int, capacity: int) -> None:
        self.index = index
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.route = []
        Vehicle.vehicle_list.append(self)
    
    def add_customer(self, customer: Customer):
        if customer.demand > self.remaining_capacity:
            raise TypeError(f'cant fit customer {customer.index} to vehicle {self.index}')
        self.remaining_capacity -= customer.demand
        self.route.insert(np.random.randint(len(self.route)+1),customer)
        
        
# class solution:
    
#     def __init__(self) -> None:
#         pass