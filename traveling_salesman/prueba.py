import random
import matplotlib.pyplot as plt
import numpy as np
import nodos

for i in range(10):
    points_array = np.random.random((4,2))
    a = nodos.Point(points_array[0][0], points_array[0][1])
    b = nodos.Point(points_array[1][0], points_array[1][1])
    c = nodos.Point(points_array[2][0], points_array[2][1])
    d = nodos.Point(points_array[3][0], points_array[3][1])
    # title = (np.linalg.norm(a-b) + np.linalg.norm(c-d) > np.linalg.norm(a-c) + np.linalg.norm(b-d)) &\
    #         (np.linalg.norm(a-b) + np.linalg.norm(c-d) > np.linalg.norm(a-d) + np.linalg.norm(b-c))

    title = nodos.intersect(a,b,c,d)

    plt.figure(figsize=(10, 10))
    plt.title(f'{title}')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.scatter(points_array[:,0], points_array[:,1])
    plt.plot(points_array[:2,0], points_array[:2,1])
    plt.plot(points_array[2:, 0], points_array[2:, 1])
    plt.show()