#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(text):
    import numpy as np
#     f = open(file=file_path)
#     text = f.read()
    lines = text.split('\n')
    lines.remove('')
    first_line = lines[0].split()
    n = int(first_line[0]) 
    K = int(first_line[1])
    
    #print('n:',n)
    #print('K:',K)
    
    lines = np.array([lines[i].split() for i in range(1,len(lines))]).astype(int)
    
    #print('Lines:\n',lines)
    
    #Planteo de solcion por dynamic programing
    
    O = np.zeros(shape=(K+1,n),dtype=int)
    
    for i in range(K+1):
        if i >= lines[0,1]:
            O[i,0] = lines[0,0]
            
    for j in range(1,n): #voy columna por columna de la tabla O
        print(np.round((j/(n-1)*100),2),'%',end='\r') #para ver el avance
        if lines[j,1] > K: #Verifico si el objeto es mas grande que la bolsa
            O[:K+1,j] = O[:K+1,j-1] #copio la columna entera
        else:
            O[:lines[j,1],j] = O[:lines[j,1],j-1] #sino copio las filas hasta uno menos que el peso del objeto
        for i in range(lines[j,1],K+1): #verifico las filas desde el peso en adelane si mejoro el valos de O
            if (O[i,j-1] > O[i-lines[j,1],j-1] + lines[j,0]):
                O[i,j] = O[i,j-1]
            elif (i - lines[j,1] >= 0):
                O[i,j] = O[i-lines[j,1],j-1] + lines[j,0]
    print('\r')
    
    traceback_k = K
    traceback_decition_x = np.zeros(n,dtype=int)
    for i in range(n-1,-1,-1):
        if O[traceback_k,i] != O[traceback_k,i-1]:
            traceback_k -= lines[i,1]
            traceback_decition_x[i] = 1
            #print('traceback_k:',traceback_k)
            #print('traceback_decition_x:',traceback_decition_x)
                
    #print(traceback_decition_x)
    
    solution = f'{O[K,n-1]} {1}\n{" ".join(list(traceback_decition_x.astype(str)))}'
    
    return solution


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

