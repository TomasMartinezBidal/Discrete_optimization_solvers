import numpy as np
import scipy.spatial as sc
from typing import List, Tuple
import copy


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

    def __repr__(self):
       return f'cust id {self.index}'
    
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
    
    def __init__(self, index: int, capacity: int, depot) -> None:
        self.index = index
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.route: List['Customer'] = [depot, depot]
        
    # def __deepcopy__(self, memo=None):
    #     return copy.deepcopy(self, memo)
    
    def add_customer(self, customer: Customer) -> Tuple[int, Customer, Customer]:
        """
        Inserts a customer into the route at a random index.

        The customer is inserted between two existing customers in the route,
        provided that their demand can be accommodated within the remaining capacity.

        Parameters:
        customer (Customer): The customer to be added to the route.

        Returns:
        Tuple[Customer, Customer, Customer]: 
            - The inserted customer
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
        return customer, self.route[inser_index-1], self.route[inser_index+1]
    
    def add_customer_at_position(self, customer:Customer, index):
        self.route.insert(index, customer)     
        self.remaining_capacity -= customer.demand
    
    def remove_customer(self, index):
        customer = self.route.pop(index)
        self.remaining_capacity += customer.demand
    
        
class Vehicle_fleet():
    
    def __init__(self, vehicle_list: List[Vehicle] = None) -> None:
        self.vehicle_list = vehicle_list
        
    def add_vehicle(self, vehicle: Vehicle):
        self.vehicle_list.append(vehicle)
        
    def __deepcopy__(self, memo=None):
        new_fleet = Vehicle_fleet()
        new_fleet.vehicle_list = [copy.deepcopy(vehicle, memo) for vehicle in self.vehicle_list]
        return new_fleet
        
    def __repr__(self):
        return f"VehicleFleet(vehicle_list={self.vehicle_list})"
    
    def find_vehicle_by_index(self, index: int) -> Vehicle:
        # Find a vehicle by its index
        for vehicle in self.vehicle_list:
            if vehicle.index == index:
                return vehicle
        raise ValueError(f"Vehicle with index {index} not found.")
        
class Solution:
    
    def __init__(self, customer_count, distance_array, distance_array_with_depot) -> None:
        self.distance_array = distance_array
        self.distance_array_with_depot = distance_array_with_depot
        self.routes = np.zeros(shape=(customer_count, customer_count))
        self.cost = 0
        self.capacity_excess = 0
        
    def add_route(self, from_customer: Customer, to_customer: Customer, array_to_modift:np.ndarray=None):
        '''Adds conection in the solution routes matrix'''
        if array_to_modift is None:
            array_to_modift = self.routes
        array_to_modift[from_customer.index, to_customer.index] = 1
        return array_to_modift
        
    def delete_route(self, from_customer: Customer, to_customer: Customer, array_to_modift:np.ndarray=None):
        '''Deletes conection in the solution routes matrix'''
        if array_to_modift is None:
            array_to_modift = self.routes
        self.routes[from_customer.index, to_customer.index] = 0
        return array_to_modift

    
    def add_customer_to_vehicle(self, vehicle: Vehicle, customer: Customer):
        '''Add a customer to a vehicle and edits the routes array'''
        inser_index, previous_customer, next_customer = vehicle.add_customer(customer)
        
        self.delete_route(previous_customer, next_customer)
        self.add_route(previous_customer, customer)
        self.add_route(customer, next_customer)
        
            
    def calc_cost(self):
        routes_array = np.copy(self.routes)
        self.cost = np.sum(self.distance_array_with_depot * routes_array)
        return self.cost
    
    def swap(self, vehicle_1: Vehicle, vehicle_2: Vehicle, customer:Customer, insertion_index_second_route=None, cost_threshold=None, capacity_threshold=None):
        '''To generate swaps we have to:
        1. select a first and a second vehicle.
        2. select a customer from the first vehicle.
        3. OPTIONAL select an index from the second vehicle.
        4.0 calculatie new cost if inserted.
        4.1 calculate capacities and total excess.
        5. OPTIONAL required maximum cost and capacity excess to generate the swap.
        '''
        # if insertion_index_second_route > len(vehicle_2.route):
        #     raise ValueError(f"index out of range for insertion {insertion_index_second_route}")
        #calculate cost if inserted
        
        # setting customer and adjacent customers in route indexes
        new_routes_array = np.copy(self.routes)
        index_customer_vehicle_1 = vehicle_1.route.index(customer)
        previous_customer = vehicle_1.route[index_customer_vehicle_1-1]
        next_customer = vehicle_1.route[index_customer_vehicle_1+1]
        
        #check for insertion index
        if insertion_index_second_route is None:
            insertion_index_second_route = np.random.randint(1, len(vehicle_2.route)-1)

        # setting new adjacent customers in second vehicle
        new_previous_customer = vehicle_2.route[insertion_index_second_route-1]
        new_next_customer = vehicle_2.route[insertion_index_second_route]
        
        #delete old routes and add new ones to new_routes_array
        self.delete_route(previous_customer, customer, new_routes_array)
        self.delete_route(customer, next_customer, new_routes_array)
        self.delete_route(new_previous_customer, new_next_customer, new_routes_array)
        self.add_route(previous_customer, next_customer, new_routes_array)
        self.add_route(new_previous_customer, customer, new_routes_array)
        self.add_route(customer, new_next_customer, new_routes_array)
        
        #calculate new cost
        new_cost = np.sum(self.distance_array_with_depot * new_routes_array)
        
        new_vehicle_1_remaining_capacity = vehicle_1.remaining_capacity + customer.demand
        new_vehicle_2_remaining_capacity = vehicle_2.remaining_capacity - customer.demand
        capacity_excess = - sum(min(cap,0) for cap in [new_vehicle_1_remaining_capacity, new_vehicle_2_remaining_capacity])
                    
        # print(f'new cost/cost_threshold: {round(new_cost)}/{round(cost_threshold)}, capacity_excess/capacity_threshold: {capacity_excess}/{capacity_threshold}, acutal cap excess: {self.capacity_excess}')
        if cost_threshold > new_cost and capacity_threshold >= capacity_excess:
            self.capacity_excess += capacity_excess# - sum(min(cap,0) for cap in [vehicle_1.remaining_capacity, vehicle_2.remaining_capacity])
            self.cost=new_cost
            self.routes = new_routes_array
            vehicle_1.remove_customer(index_customer_vehicle_1)
            vehicle_2.add_customer_at_position(customer, insertion_index_second_route)
            # print(self.routes, new_routes_array, sep='\n')
            # print(self.cost, self.calc_cost())
            # print(f'cost: {self.cost}')
            return True
        return False
        
        
    
    def modify(self, customer, vehicle, new_position):
        
        pass