import matplotlib.pyplot as plt

def plot_solution(customers, solution=None):
    
    fig, ax = plt.subplots()

    x_values = [customer.x for customer in customers]
    y_values = [customer.y for customer in customers]

    ax.scatter(0, 0)
    ax.scatter(x_values, y_values)

    plt.show()
    
    return fig