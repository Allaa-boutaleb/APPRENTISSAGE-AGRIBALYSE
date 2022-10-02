# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2022

# import externe
from collections import defaultdict
import numpy as np
import scipy.cluster.hierarchy
import pandas as pd
import copy
import math

import matplotlib.pyplot as plt

def normalisation(df) :
    return (df - df.min())/(df.max() - df.min())

def dist_euclidienne(vect1, vect2) :
    return np.linalg.norm(vect1-vect2)
    
    
def dist_manhattan(v1,v2) :
    n = len(v1)
    sum = 0
    for i in range(n) :
        sum += abs(v1[i] - v2[i])
    return sum
        

def dist_vect(v1, v2):
    return dist_euclidienne(v1,v2)

def inertie_cluster(Ens):
    distances = 0
    centroid = centroide(Ens)
    for x in range(len(Ens)) :
        distances += (dist_euclidienne(Ens.iloc[x],centroid)**2)
    return distances


def inertie_globale(Base, U):
    result = 0
    for key in U.keys() :
        result +=inertie_cluster(Base.iloc[U[key]])
    return result

def nouveaux_centroides(Base,U):
    centroides = []
    for key in U.keys() :
        centroides.append(Base.iloc[U[key]].mean(axis=0))
    return np.asarray(centroides)
    

def kmoyennes(K, Base, epsilon, iter_max):

    n_iterations = 0
    init_centroides = init_kmeans(K,Base)
    matrice = affecte_cluster(Base,init_centroides)
    inertie_prec = inertie_globale(Base,matrice)
    inertie_current = inertie_prec + epsilon + 1

    while (abs(inertie_prec - inertie_current) > epsilon) and (n_iterations < iter_max) :
        nv_centroides = nouveaux_centroides(Base,matrice)
        matrice = affecte_cluster(Base,nv_centroides)
        inertie_prec = inertie_current
        inertie_current = inertie_globale(Base,matrice)

        n_iterations += 1
        

    return nv_centroides, matrice

def centroide(df) :
    return df.mean(axis=0)
              
def codistance(Base, Centres, Affect) :
    # Calcul de codistance
    dk_list = []
    for i in range(len(Centres)) :
        tmp = Base.iloc[Affect[i]].values
        max_dist = 0

        for e1 in tmp :
            for e2 in tmp :
                tmp2 = dist_euclidienne(e1,e2)
                if tmp2 > max_dist and tmp2 != 0:
                    max_dist = tmp2
        dk_list.append(max_dist)

    codistance = max(dk_list)

    return codistance


def sepmin(Base,Centres,Affect) :
    # ----------------- Calcul de sepmin
    min_dist = float('inf')
    for c1 in Centres :
        for c2 in Centres :
            tmp = dist_euclidienne(c1,c2)
            if tmp < min_dist and (tmp != 0) :
                min_dist = tmp
    return min_dist

def index_Dunn(Base,Centres,Affect) :  
    return  sepmin(Base,Centres,Affect) / codistance(Base,Centres,Affect)     

def index_xie_beni(Base,Centres,Affect) :
    return inertie_globale(Base,Affect) / sepmin(Base,Centres,Affect)

def init_kmeans(K,Ens):
    indexes = np.random.choice(Ens.shape[0],K,replace=False)
    return np.array(Ens.iloc[indexes])

def plus_proche(Exe,Centres):
    distances = [dist_euclidienne(Exe,Centres[i]) for i in range(len(Centres))]
    return distances.index(min(distances))

def centroide(df) :
    return df.mean(axis=0)
        
def dist_centroides(v1,v2) :
    return dist_euclidienne(centroide(v1), centroide(v2))

# Autres distances
def dist_complete(v1,v2) :
    distances = []
    for x in range(len(v1)) :
        for y in range(len(v2)) :
            distances.append(dist_euclidienne(v1.iloc[x],v2.iloc[y]))
    return max(distances)

def dist_simple(v1,v2) :
    distances = []
    for x in range(len(v1)) :
        for y in range(len(v2)) :
            distances.append(dist_euclidienne(v1.iloc[x],v2.iloc[y]))
    return min(distances)

def dist_average(v1,v2) :
    distances = []
    for x in range(len(v1)) :
        for y in range(len(v2)) :
            distances.append(dist_euclidienne(v1.iloc[x],v2.iloc[y]))
    return sum(distances) / len(distances)


def initialise(df) :
    dict1 = dict() 
    
    for i in range(len(df)):
        dict1[i] = [i]
        
    return dict1


def fusionne(df, start, verbose=False, linkage="centroide") :
    
    newclusters = copy.deepcopy(start)
    
    distances = []
    index1 = []
    index2 = []
    
    i1 = 0
    i2 = 0
    minimum = 0
    
    for i in start.keys() :
        for j in start.keys() :
            x1 = df.iloc[start[i]]
            x2 = df.iloc[start[j]]
            if i != j :
                if linkage == "centroide" :
                    tmp = dist_centroides(df.iloc[start[i]],df.iloc[start[j]])
                elif linkage == "complete" :
                    tmp = dist_complete(df.iloc[start[i]],df.iloc[start[j]])
                elif linkage == "simple" :
                    tmp = dist_simple(df.iloc[start[i]],df.iloc[start[j]])
                else : #average
                    tmp = dist_average(df.iloc[start[i]],df.iloc[start[j]])
                distances.append(tmp)
                index1.append(i)
                index2.append(j)
    
    minimum = distances.index(min(distances))
    i1 = index1[minimum]
    i2 = index2[minimum]
    
    # -------------------------------------------
    del newclusters[i1]
    del newclusters[i2]
    
    lastkey = sorted(start.keys())[-1]
    newclusters[lastkey+1] = start[i1]+start[i2]
    
    if verbose :
        print("Distance minimale trouvée entre [",start[i1],",",start[i2],"] = ", min(distances))
    return newclusters, i1, i2, min(distances)

def affecte_cluster(Base,Centres):
    matrice = {new_list: [] for new_list in range(len(Centres))}
    for x in range(len(Base)) :
        matrice[plus_proche(Base.iloc[x],Centres)].append(x)

    return matrice
        

def clustering_hierarchique(df, verbose=False, dendrogramme=False, linkage="centroide") :
    results = []
    depart = initialise(df)

    while(len(depart) != 1) :
        depart,i1,i2,m = fusionne(df,depart,verbose,linkage)
        results.append([i1,i2,m,len(depart[sorted(depart.keys())[-1]])])



    if (dendrogramme) : 
        
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            results, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    
    return results

def affiche_resultat(Base,Centres,Affect):
    colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray"])
    for i in range(len(Centres)) :
        plt.scatter(Base.iloc[Affect[i]][0],Base.iloc[Affect[i]][1], c=colors[i])
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
        
        
def clustering_info(df, results, nbr_clusters, col) : 
    arr = scipy.cluster.hierarchy.fcluster(results, nbr_clusters, 'maxclust')
    lst = []

    for i in range(len(arr)) :
        lst.append((arr[i],df.iloc[i,col]))

    clusters = defaultdict(list)

    # iterating over list of tuples
    for key, val in lst:
        clusters[key].append(val)
        
    for key in range(1,nbr_clusters+1) :
        values,counts = np.unique(clusters[key],return_counts=True)
        print("- Cluster ", key, " : ", values, " - ", counts)


        
        
        
        
    
            
    