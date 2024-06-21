import numpy as np
import scipy.spatial as sc
from typing import List


class Customer:
    customer_list: List['Customer'] = []
    customer_list_by_demand : List['Customer'] = []
    y_coords = np.array([])
    x_coords = np.array([])
    distances: np.ndarray
    distances_with_depot: np.ndarray
    total_nodes = 0
    depot = None 
    
    def __init__(self, index : int = None, demand : int = None, x : float = None, y : float = None, depot: bool= False, customer_count=None) -> None:
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
        if depot:
            self.__class__.depot = self
            self.index = customer_count
    
    @classmethod
    def clac_distances(cls): 
        array_x_y = np.array((cls.x_coords, cls.y_coords)).T
        cls.distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
        cls.ordered_customers_distance = np.argsort(cls.distances)
        for i, customer in enumerate(cls.customer_list):
            customer.distance_to_customers = cls.distances[i]
            customer.ordered_customers_distance = cls.distances[i]
        
        array_x_y_with_depot = np.array(([cls.depot.x]+cls.x_coords, [cls.depot.y]+cls.y_coords))        
        cls.distances_with_depot = sc.distance.squareform(sc.distance.pdist(array_x_y_with_depot))

    @classmethod
    def generate_customer_list_by_demand(cls):
        cls.customer_list_by_demand = sorted(cls.customer_list, key=lambda customer: customer.demand, reverse=True)


        
class Vehicle:
    cost = 0
    vehicle_list: List['Vehicle'] = []
    
    def __init__(self, index: int, capacity: int, depot) -> None:
        self.index = index
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.route: List['Customer'] = [depot, depot]
        Vehicle.vehicle_list.append(self)
    
    def add_customer(self, customer: Customer):
        #print('adding', customer.index, customer.demand, self.index, self.remaining_capacity)
        if customer.demand > self.remaining_capacity:
            raise TypeError(f'cant fit customer {customer.index} to vehicle {self.index}')
        self.remaining_capacity -= customer.demand
        inser_index = np.random.randint(1, len(self.route))
        self.route.insert(inser_index, customer)
        #print('adding', customer.index, customer.demand, self.index, self.remaining_capacity)
        return inser_index
        
        
class Solution:
    
    def __init__(self, customer_count, distance_array) -> None:
        self.distance_array = distance_array
        self.coming_from_array = np.zeros(shape=(customer_count+1, customer_count+1))
        self.going_to_array = np.zeros(shape=(customer_count+1, customer_count+1))
        self.cost = 0
        pass
    
    def calc_cost(self):
        print(self.distance_array, self.used_routes, sep="\n")
        self.cost = np.sum(self.distance_array * self.used_routes)
        return self.cost
        
    def add(self, vehicle: Vehicle, customer: Customer):
        inser_index = vehicle.add_customer(customer)
        from_customer = vehicle.route[inser_index - 1]
        to_customer = vehicle.route[inser_index + 1]
        
        self.coming_from_array[to_customer.index, from_customer.index] = 0
        self.going_to_array[from_customer.index, to_customer.index] = 0
        
        self.coming_from_array[customer.index, from_customer.index] = 1
        self.going_to_array[customer.index, to_customer.index] = 1
            
    def modify(self, customer, vehicle, new_position):
        
        pass