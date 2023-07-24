#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(file_path):
    import numpy as np
    #f = open(file=file_path)
    #text = f.read()
    text = file_path #eliminar y descomentar las anteriores
    lines = text.split('\n')
    lines.remove('') #descoemntar tambien
    first_line = lines[0].split()
    n = int(first_line[0]) 
    K = int(first_line[1])
    
    print('n:',n)
    print('K:',K)
    if n > 399:
        print('greedy')
        lines = np.array([lines[i].split() for i in range(1,len(lines))]).astype(int)

        #print(lines)

        lines = np.append(lines,(lines[:,0]/lines[:,1]).reshape(-1,1),axis=1)

        sorted_val = np.argsort(lines[:,0])[::-1]
        sorted_wei = np.argsort(lines[:,1])
        sorted_eff = np.argsort(lines[:,2])[::-1]
        sorted_val_eff = np.lexsort((lines[:,2],lines[:,0]))[::-1]
        sorted_eff_val = np.lexsort((lines[:,0],lines[:,2]))[::-1]
        sorted_val_wei = np.lexsort((-lines[:,1],lines[:,0]))[::-1]
        sorted_eff_wei = np.lexsort((-lines[:,1],lines[:,2]))[::-1]

        sorts = [sorted_val,sorted_wei,sorted_eff,sorted_val_eff,sorted_eff_val,sorted_val_wei,sorted_eff_wei]

        values = np.array([])
        responses = np.array([])

        for i in sorts:

            cargo = 0
            value = 0
            sorted_lines = lines[i]
            response = np.zeros(n,dtype=int)

            for j in range(n):
                if cargo + sorted_lines[j,1] < K:
                    cargo += sorted_lines[j,1]
                    value += sorted_lines[j,0]
                    response[i[j]] = 1
            response = response.astype(int)
            values = np.append(values,value)
            responses = np.append(responses,response)

        responses = np.reshape(responses,(len(sorts),-1))

        index = np.argmax(values)

        value = values[index]
        response = responses[index]

        solution = f'{value.astype(int)} {0}\n{" ".join(list(response.astype(int).astype(str)))}'

        return solution
    else:
        print('solver O')
        lines = np.array([lines[i].split() for i in range(1,len(lines))]).astype(int)

        #print('Lines:\n',lines)

        #Planteo de solcion por dynamic programing

        O = np.zeros(shape=(K+1,n),dtype=int)

        for i in range(K+1):
            if i >= lines[0,1]:
                O[i,0] = lines[0,0]

        for j in range(1,n): #voy columna por columna de la tabla O
            #print(j/(n-1)*100,end='\r') #para ver el avance
            if lines[j,1] > K: #Verifico si el objeto es mas grande que la bolsa
                O[:K+1,j] = O[:K+1,j-1] #copio la columna entera
            else:
                O[:lines[j,1],j] = O[:lines[j,1],j-1] #sino copio las filas hasta uno menos que el peso del objeto
            for i in range(lines[j,1],K+1): #verifico las filas desde el peso en adelane si mejoro el valos de O
                if (O[i,j-1] > O[i-lines[j,1],j-1] + lines[j,0]):
                    O[i,j] = O[i,j-1]
                elif (i - lines[j,1] >= 0):
                    O[i,j] = O[i-lines[j,1],j-1] + lines[j,0]

    #     print(O)

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

        #print(np.append(lines, traceback_decition_x.reshape(-1,1),axis=1))
        
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

