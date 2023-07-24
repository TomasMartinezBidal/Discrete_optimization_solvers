import pandas as pd
import scipy.spatial as sc
import numpy as np
from plot_solution_file import plot_solution

def make_facilities_customers_df(lines, facility_count, customer_count):
    # Function to create two datadrames whith the information of facilities and customers.
    facilities = pd.DataFrame(columns=['setup_cost', 'capacity', 'x', 'y'])
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.loc[i - 1] = parts
    facilities = facilities.astype({'setup_cost':float, 'capacity':int, 'x':float, 'y':float})
    print('done facilities')
    customers = pd.DataFrame(columns=['demand', 'x', 'y'])
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.loc[i - 1 - facility_count] = parts
    customers = customers.astype({'demand':int, 'x':float, 'y':float})
    print('done customers')

    return facilities, customers

def calc_distances(facility_count, customer_count, facilities, customers):
    array_x_y = pd.concat([facilities.loc[:,['x','y']], customers.loc[:,['x','y']]])

    distances = sc.distance.squareform(sc.distance.pdist(array_x_y))
    distances_facilities = distances[0:facility_count, 0:facility_count]
    distances_customers = distances[facility_count:, facility_count:]
    distances_facilities_customers = distances[facility_count:, :facility_count]  # Customer in row, facility in column

    ordered_distances_facilities = np.argsort(distances_facilities, axis=0)
    ordered_distances_customers = np.argsort(distances_customers, axis=0)
    ordered_rowfacility_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=0)
    ordered_rowcustomer_distances_facilities_customers = np.argsort(distances_facilities_customers, axis=1)

    return distances, distances_facilities, distances_customers, distances_facilities_customers, ordered_distances_facilities,\
        ordered_distances_customers, ordered_rowfacility_distances_facilities_customers,\
        ordered_rowcustomer_distances_facilities_customers
def gen_decition_df(facility_count, customer_count, facilities):
    # Function to create decition variable arrays
    x = pd.DataFrame(np.zeros((customer_count, facility_count), dtype='int'))  # df para las decition variables

    f = pd.DataFrame(np.zeros(facility_count, dtype='int'), columns=['facility_open'])

    return f, x

def greedy_solver(customer_count, facilities, customers, ordered_rowcustomer_distances_facilities_customers, x ,f,
                  greedy_facility_maks = None, plot = False):

    # Function to create a greedy solution

    facility_capacity_series = facilities.loc[:,'capacity'].copy()
    f.facility_open = 1

    for customer in range(customer_count):
        if greedy_facility_maks:  # If statement to apply greedy mask if passed.
            open_facilities = greedy_facility_maks  # In case I want to restrict the available facilities.
        else:
            open_facilities = np.ones(facilities.shape[0], dtype=bool)
        open_facilities_w_capacity_maks = (facility_capacity_series > customers.loc[customer,'demand']) & open_facilities
        open_facilities_w_capacity_dist_ordered = ordered_rowcustomer_distances_facilities_customers[customer][open_facilities_w_capacity_maks[ordered_rowcustomer_distances_facilities_customers[customer]]]  # Index list of facilities!
        facility_to_be_assigned = open_facilities_w_capacity_dist_ordered[0]
        facility_capacity_series.loc[facility_to_be_assigned] -= customers.loc[customer, 'demand']  # Update capacity
        x.loc[customer, facility_to_be_assigned] = 1  # Open the facility in x.

    # Close all facilities that are not occupied.
    mask_close_facilities = facility_capacity_series == facilities.capacity
    mask_open_facilities = ~mask_close_facilities
    f.loc[mask_close_facilities, 'facility_open'] = 0

    facility_mean_capacity = facilities.loc[:, 'capacity'].mean()
    facility_mean_setup_cost = facilities.loc[:, 'setup_cost'].mean()
    extra_facilities_mask = (facilities.loc[:, 'capacity'] >= facility_mean_capacity) & (
                facilities.loc[:, 'setup_cost'] <= facility_mean_setup_cost)
    extra_facilities_mask = extra_facilities_mask & mask_close_facilities
    mask_open_facilities_pulp = mask_open_facilities | extra_facilities_mask

    if plot:  # Save an image of the solution plotted
        hue_array = (mask_open_facilities).map({True:'Open', False:'Close'})
        facility_mean_capacity = facilities.loc[:,'capacity'].mean()
        facility_mean_setup_cost =  facilities.loc[:,'setup_cost'].mean()
        # if facility_count <= 100 & customer_count <= 1000:
        #     facility_mean_capacity = 0
        #     facility_mean_setup_cost = 0
        extra_facilities_mask = (facilities.loc[:,'capacity'] >= facility_mean_capacity) & (facilities.loc[:,'setup_cost'] <= facility_mean_setup_cost)
        extra_facilities_mask = extra_facilities_mask & mask_close_facilities
        hue_array[extra_facilities_mask] = 'New_open'
        # g = sns.jointplot(data=facilities, x='setup_cost', y='capacity', hue=hue_array)
        # plt.show()

        mask_open_facilities_pulp = mask_open_facilities | extra_facilities_mask

        # decition_variables_array = np.array([ i if isinstance(i,np.int32) else i.varValue for i in x.values.flatten()])
        # decition_variables_array = decition_variables_array.reshape((customer_count, facility_count))
        solution = list(x.values.argsort()[:, -1])
        print(f'Greedy solution:\n{solution}')
        plot_solution(facilities=facilities, customers=customers, solution=solution, extra='greedy')

    return mask_open_facilities, mask_close_facilities, mask_open_facilities_pulp

def efficiency_mask_generator(facility_count, facilities, customers, ordered_rowfacility_distances_facilities_customers, distances_facilities_customers):
    # The capacity factor is set to 4, should be changed to have more effect, but it results in slower pulp solving time.

    for facility_n in range(facility_count):
        facility_capacity = facilities.loc[facility_n, "capacity"] * 1

        customers_sort_order_dist = ordered_rowfacility_distances_facilities_customers[:, facility_n]

        customers_distances = distances_facilities_customers[:, facility_n]

        customers_demands = customers.loc[:, "demand"].values

        customers_demand_cost = customers_demands / customers_distances

        customers_sort_order_demand_cost = np.argsort(customers_demand_cost)[::-1]

        cost_array = np.array(facilities.loc[facility_n, "setup_cost"])
        cost_array = np.append(arr=cost_array, values=customers_distances[customers_sort_order_demand_cost]).cumsum()
        demand_array = np.array(0)
        demand_array = np.append(arr=demand_array, values=customers_demands[customers_sort_order_demand_cost]).cumsum()

        demand_mask = demand_array <= facility_capacity

        demand_array = demand_array[demand_mask]
        cost_array = cost_array[demand_mask]

        facilities.loc[facility_n, "eficiency"] = facilities.loc[facility_n, "capacity"] / cost_array[-1]

        facilities_use_mask = facilities.loc[:, "eficiency"]

    facilities.sort_values(by='eficiency', ascending=False, inplace=True)  # Order facilities by eficiency.
    facilities.loc[:, 'cumulative_cap_ef'] = np.cumsum(facilities.loc[:, 'capacity'])  # Generate a cumulative column.
    capacity_req = customers.demand.sum() * 4  # Define a required capacity bigger than actual capacity requirement.
    print(f'Efficiency total capacity reduction: {capacity_req / facilities.capacity.sum()}')
    facilities_ef_mask = facilities.cumulative_cap_ef < capacity_req  # Generate a mask of facilities selected by efficiency. Its a series.
    facilities_ef_mask.sort_index(inplace=True)
    facilities.sort_index(inplace=True)
    available_facilities = pd.Series(range(facility_count))[facilities_ef_mask].values  # List of facilities
    mask_open_facilities_pulp = facilities_ef_mask

    return available_facilities, mask_open_facilities_pulp
