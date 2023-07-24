import numpy as np
import matplotlib.pyplot as plt
import inspect


def plot_solution(facilities, customers, file_location = 'data\\fl_x_x', solution=[] ,extra=''):

    fig, ax = plt.subplots(nrows=1,ncols=1)
    ax.scatter(facilities.x, facilities.y)
    ax.scatter(customers.x, customers.y)
    for index, row in facilities.iterrows():
        plt.annotate(f"f{index} \n {row.loc[['x','y']]}", row.loc[['x','y']])#, fontsize=2)
    for index, row in customers.iterrows():
        plt.annotate(f"c{index} \n {row.loc[['x','y']]}", row.loc[['x','y']])#, fontsize=2)
    if len(solution) > 0:
        for i in range(len(solution)):
            customer_xy = customers.loc[[i],['x', 'y']].values
            facility_xy = facilities.loc[[solution[i]], ['x', 'y']].values
            coords = np.concatenate([customer_xy,facility_xy])
            ax.plot(coords[:,0], coords[:,1], c='g')

    caller = inspect.getouterframes(inspect.currentframe(), 2)[1][1].split('.')[0].split('\\')[-1]
    # print(caller)

    # name = file_location.split('\\')[1]
    name = f'{"_".join(str(x) for x in [caller,facilities.shape[0],customers.shape[0], extra])}'
    fig.suptitle(name)
    plt.savefig(f'images\{name}.png')#, dpi=1000)
    return None