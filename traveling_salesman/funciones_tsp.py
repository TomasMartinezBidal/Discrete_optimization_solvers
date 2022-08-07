import numpy as np

def dist_path(indexes, x_y_array):  # funcion para calcular la distancia de un camino propuesto
    lenght = 0
    for i in range(len(indexes) - 1):
        x1 = x_y_array[indexes[i], 0]
        x2 = x_y_array[indexes[i + 1], 0]
        y1 = x_y_array[indexes[i], 1]
        y2 = x_y_array[indexes[i + 1], 1]
        lenght += np.linalg.norm([x1 - x2, y1 - y2])
    return lenght

def nearest_points(indexes_1, indexes_2, x_y_array):  # funcion para obtener los dos puntos mas cercano de dos grupos distintos
    dist = np.inf
    for i in range(len(indexes_1)):
        for j in range(len(indexes_2)):
            x1 = x_y_array[indexes_1[i], 0]
            x2 = x_y_array[indexes_2[j], 0]
            y1 = x_y_array[indexes_1[i], 1]
            y2 = x_y_array[indexes_2[j], 1]
            if dist > np.linalg.norm([x1 - x2, y1 - y2]):
                dist = np.linalg.norm([x1 - x2, y1 - y2])
                nearest_i = indexes_1[i]
                nearest_j = indexes_2[j]
    return (nearest_i, nearest_j)