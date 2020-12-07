import csv
import numpy as np

def load_data(filepath):
    '''takes in a string with a path to a CSV file, 
       returns the first 20 data points (without the Generation and Legendary columns 
       but retaining all other columns) in a single structure.'''
    dataset = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row_num = 1
        for row in reader:
            if row_num > 20:
                break
            del row['Generation']
            del row['Legendary']
            row['HP'] = int(row['HP'])
            row['Attack'] = int(row['Attack'])
            row['Defense'] = int(row['Defense'])
            row['Sp. Atk'] = int(row['Sp. Atk'])
            row['Sp. Def'] = int(row['Sp. Def'])
            row['Speed'] = int(row['Speed'])
            row['#'] = int(row['#'])
            row['Total'] = int(row['Total'])
            dataset.append(row)
            row_num += 1
    return dataset

def calculate_x_y(stats):
    '''takes in one row from the data loaded from the previous function,
       calculates the corresponding x, y values for that Pokemon,
       returns them in a single structure'''
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return (x,y)

def distance(clst_1, clst_2):
    '''computes the euclidean distance between two clusters/points'''
    ret = 10000000
    for p1 in clst_1:
        for p2 in clst_2:
            dis = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            if dis < ret:
                ret = dis
    return ret

def delete_clst(clst_list, index):
    '''performs deletion of a cluster entry from list based on the given index'''
    for i in range(len(clst_list)):
        if clst_list[i][1] == index:
            break
    del clst_list[i]
    return clst_list

def hac(dataset):
    '''performs single linkage hierarchical agglomerative clustering on the Pokemon
       with the (x,y) feature representation, and returns a data structure representing the clustering.'''
    z = []
    cur_clst = []
    all_clst = []
    max_index = len(dataset) - 1
    for i in range(len(dataset)):
        temp = []
        temp.append([dataset[i]])
        temp.append(i)
        cur_clst.append(temp)
    while len(cur_clst) > 1:
        min_dis = 10000000;
        clst_1 = []
        clst_2 = []
        index_1 = -1
        index_2 = -1
        # computer min dis between any two clusters, save the cluster index
        for i in range(len(cur_clst)):
            # second loop start after i so that i always smaller than j
            for j in range(i+1,len(cur_clst)):
                dis = distance(cur_clst[i][0],cur_clst[j][0])
                if dis < min_dis:
                    min_dis = dis
                    clst_1 = cur_clst[i][0]
                    clst_2 = cur_clst[j][0]
                    index_1 = cur_clst[i][1]
                    index_2 = cur_clst[j][1]
        # we now combine cluster at index i and j to form a new cluster
        comb = []    
        for i in range(len(clst_1)):
            comb.append(clst_1[i])
        for i in range(len(clst_2)):
            comb.append(clst_2[i])
        # add new cluster to list
        max_index += 1
        temp = []
        temp.append(comb)
        temp.append(max_index)
        cur_clst.append(temp)
        # add the merge info to z
        z.append([index_1,index_2, min_dis, len(comb)])
        # delecte the two clusters that made up the new one from cur_clst
        cur_clst = delete_clst(cur_clst, index_1)
        cur_clst = delete_clst(cur_clst, index_2)
    return np.asmatrix(z)
