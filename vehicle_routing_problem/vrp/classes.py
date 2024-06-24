import numpy as np
import scipy.spatial as sc
from typing import List, Tuple


class Customer:
    customer_list: List['Customer'] = []
    customer_list_by_demand : List['Customer'] = []
    y_coords = np.array([])
    x_coords = np.array([])
    distances: np.ndarray
    distances_with_depot: np.ndarray
    total_nodes = 0
    depot = None 
    
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
        if depot:
            self.__class__.depot = self
            self.index = 0
    
    @classmethod
    def clac_distances(cls): 
        array_x_y = np.array((cls.x_coords, cls.y_coords)).T
        cls.distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
        cls.ordered_customers_distance = np.argsort(cls.distances)
        for i, customer in enumerate(cls.customer_list):
            customer.distance_to_customers = cls.distances[i]
            customer.ordered_customers_distance = cls.distances[i]
        
        array_x_y_with_depot = np.array((np.append([cls.depot.x], cls.x_coords), np.append([cls.depot.y], cls.y_coords))).T       
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
    
    def add_customer(self, customer: Customer) -> Tuple[int, Customer, Customer]:
        """
        Inserts a customer into the route at a random index.

        The customer is inserted between two existing customers in the route,
        provided that their demand can be accommodated within the remaining capacity.

        Parameters:
        customer (Customer): The customer to be added to the route.

        Returns:
        Tuple[int, Customer, Customer]: 
            - The index at which the customer was inserted.
            - The customer before the inserted customer in the route.
            - The customer after the inserted customer in the route.

        Raises:
        TypeError: If the customer's demand exceeds the remaining capacity of the vehicle.
        """
        if customer.demand > self.remaining_capacity:
            raise TypeError(f'cant fit customer {customer.index} to vehicle {self.index}')
        self.remaining_capacity -= customer.demand
        inser_index = np.random.randint(1, len(self.route))
        self.route.insert(inser_index, customer)
        return inser_index, self.route[inser_index-1], self.route[inser_index+1]
        
        
class Solution:
    
    def __init__(self, customer_count, distance_array, distance_array_with_depot) -> None:
        self.distance_array = distance_array
        self.distance_array_with_depot = distance_array_with_depot
        self.routes = np.zeros(shape=(customer_count, customer_count))
        # self.coming_from_array = np.zeros(shape=(customer_count, customer_count))
        # self.going_to_array = np.zeros(shape=(customer_count, customer_count))
        self.cost = 0
        
    def add_route(self, customer_1: Customer, customer_2: Customer):
        '''Adds conection in the solution routes matrix'''
        min_index, max_index = sorted((customer_1.index, customer_2.index))
        self.routes[min_index, max_index] = 1
        
    def delete_route(self, customer_1: Customer, customer_2: Customer):
        '''Deletes conection in the solution routes matrix'''
        min_index, max_index = sorted((customer_1.index, customer_2.index))
        self.routes[min_index, max_index] = 0
    
    def add_to_vehicle(self, vehicle: Vehicle, customer: Customer):
        '''Add a customer to a vehicle and edits the routes array'''
        inser_index, previous_customer, next_customer = vehicle.add_customer(customer)
        
        self.delete_route(previous_customer, next_customer)
        self.add_route(previous_customer, customer)
        self.add_route(customer, next_customer)
        
            
    def calc_cost(self):
        routes_array = np.copy(self.routes)
        self.cost = np.sum(self.distance_array_with_depot * routes_array)
        return self.cost
    
    def modify(self, customer, vehicle, new_position):
        
        pass